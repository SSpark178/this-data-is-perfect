"""Adaptive, budget-safe recovery of a sparse bilinear function with a hard halfspace step."""
import itertools
import math

import numpy as np

VARS = [f'x{i}' for i in range(1, 6)]


def solve(oracle, points):
    data = {}

    def key(x):
        return tuple(round(float(v), 9) for v in x)

    def ask(x, phase='probe'):
        x = [min(1.0, max(0.0, float(v))) for v in x]
        k = key(x)
        if k not in data:
            if oracle.remaining < 1:
                raise RuntimeError('Internal query budget exhausted')
            data[k] = float(oracle.ask(x, phase=phase))
        return data[k]

    def ask_all(xs, phase):
        return [ask(x, phase) for x in xs]

    # Probe two tiny factorial neighborhoods.  At least one will normally be
    # entirely on one side of the discontinuity, allowing clean algebraic
    # recovery of the continuous polynomial without assumptions on its pair.
    def patch(base, delta):
        p0 = [float(base)]*5 if isinstance(base,(int,float)) else list(base)
        ds = [float(delta)]*5 if isinstance(delta,(int,float)) else list(delta)
        y0 = ask(p0, 'local polynomial')
        singles = {}
        for i in range(5):
            p = p0.copy(); p[i] += ds[i]
            singles[i] = ask(p, 'local polynomial')
        pair = np.zeros((5, 5))
        for i in range(5):
            for j in range(i+1, 5):
                p = p0.copy(); p[i] += ds[i]; p[j] += ds[j]
                yij = ask(p, 'local polynomial')
                pair[i, j] = (yij - singles[i] - singles[j] + y0)/(ds[i]*ds[j])
        grad = np.array([(singles[i]-y0)/ds[i] for i in range(5)])
        b = np.array(p0)
        slopes = grad.copy()
        for i in range(5):
            for j in range(i+1, 5):
                slopes[i] -= pair[i, j]*b[j]
                slopes[j] -= pair[i, j]*b[i]
        intercept = y0 - float(slopes@b) - sum(pair[i, j]*b[i]*b[j]
                                      for i in range(5) for j in range(i+1, 5))
        return intercept, slopes, pair

    patch_low = patch(.10, .12)
    patch_high = patch(.90, -.12)
    corners = [tuple(v) for v in itertools.product((0.0, 1.0), repeat=5)]
    ask_all(corners, 'factorial corners')
    # Independent interior observations make the two-level residual check
    # reject local patches that accidentally cross the jump surface.
    rng = np.random.default_rng(2026)
    interior = [tuple(v) for v in rng.uniform(.03,.97,(14,5))]
    ask_all(interior, 'interior validation')

    def polyeval(model, x):
        b, a, pair = model
        return float(b+sum(a[i]*x[i] for i in range(5))+
                     sum(pair[i,j]*x[i]*x[j] for i in range(5)
                         for j in range(i+1, 5)))

    def split_residuals(rs):
        vals = np.array(sorted(rs), dtype=float)
        n = len(vals)
        ss = np.cumsum(vals)
        ss2 = np.cumsum(vals*vals)
        best = (float('inf'), 1)
        for k in range(1, n):
            n0, n1 = k, n-k
            s0, s1 = ss[k-1], ss[-1]-ss[k-1]
            q0, q1 = ss2[k-1], ss2[-1]-ss2[k-1]
            score = max(0.0, q0-s0*s0/n0) + max(0.0, q1-s1*s1/n1)
            if score < best[0]:
                best = (score, k)
        idx = best[1]
        return best[0], float(np.mean(vals[:idx])), float(np.mean(vals[idx:]))

    def candidate_quality(model):
        res = [ask(x)-polyeval(model, x) for x in corners+interior]
        mse, left, right = split_residuals(res)
        # In an unbroken patch there is precisely one interaction. A step
        # crossing the patch often creates several fake pairwise terms.
        nonzero = sorted([abs(float(model[2][i,j])) for i in range(5)
                          for j in range(i+1,5)], reverse=True)
        return mse + min(10.0, nonzero[1]**2)*.00001, left, right

    qlow = candidate_quality(patch_low)
    qhigh = candidate_quality(patch_high)
    poly = patch_low if qlow[0] <= qhigh[0] else patch_high
    _, lowres, highres = qlow if qlow[0] <= qhigh[0] else qhigh
    # If neither local factorial was inside a single phase, search for a
    # corner whose entire 1-hop neighborhood shares one observed phase.
    if min(qlow[0], qhigh[0]) > .005:
        # An unstable local patch generates a spurious, enormous cross term
        # on the gate pair (jump / delta**2).  Place the next factorial patch
        # at a *mixed* corner of that pair.  This is especially important
        # when the gate weights have opposite signs: both diagonal corners
        # can lie right on the step, but mixed corners lie far from it.
        diffs=[]
        for i in range(5):
            for j in range(i+1,5):
                ca=float(patch_low[2][i,j]);cb=float(patch_high[2][i,j])
                diffs.append((abs(ca-cb),max(abs(ca),abs(cb)),i,j))
        diffs.sort(reverse=True)
        if diffs[0][0]>5.0:
            gi,gj=diffs[0][2:]
        else:
            gi,gj=max(diffs,key=lambda z:z[1])[2:]
        corner = tuple(1.0 if z==gj else 0.0 for z in range(5))
        third = patch([.04 if z==0 else .96 for z in corner],
                      [.10 if z==0 else -.10 for z in corner])
        qthird = candidate_quality(third)
        if qthird[0]<min(qlow[0],qhigh[0]):
            poly=third
            _,lowres,highres=qthird
    jump0 = float(highres-lowres)
    middle = (lowres+highres)/2

    def label(x):
        return int(ask(x)-polyeval(poly, x) > middle)

    corner_labels = {p:label(p) for p in corners}
    axes = {}
    for i in range(5):
        edges = []
        for p0 in corners:
            if p0[i] != 0.0:
                continue
            p1 = list(p0); p1[i] = 1.0; p1 = tuple(p1)
            if corner_labels[p0] != corner_labels[p1]:
                edges.append((p0,p1))
        if edges:
            # For positive-weight gates, all-one backgrounds tend to give
            # interior crossings rather than a threshold right at an endpoint.
            edges.sort(key=lambda edge: sum(edge[0]), reverse=True)
            axes[i] = edges

    def bisect(p0, p1, n=12):
        lo, hi = list(p0), list(p1)
        la = label(lo)
        assert la != label(hi)
        for _ in range(n):
            mid = [(a+b)/2 for a,b in zip(lo,hi)]
            if label(mid)==la:
                lo=mid
            else:
                hi=mid
        return [(a+b)/2 for a,b in zip(lo,hi)]

    boundary_pts = []
    gate_axes = sorted(axes)
    if gate_axes:
        main_axis = gate_axes[0]
        main_edge = axes[main_axis][0]
        # If several crossing edges for this axis exist, prefer an interior
        # intercept by checking three midpoints, limited to a few queries.
        if len(axes[main_axis]) > 1:
            tried=[]
            for edge in [axes[main_axis][0],axes[main_axis][-1]]:
                mid=list(edge[0]); mid[main_axis]=.5
                lm=label(mid)
                # 0.5 is less desirable than a firmly interior threshold,
                # but either gives a valid bisection interval.
                tried.append((lm,edge))
            main_edge=tried[0][1]
        p_main = bisect(*main_edge)
        boundary_pts.append(p_main)
        if len(gate_axes) > 1:
            # Any gate coefficient that changes a vertex label is guaranteed
            # to create such an edge.  A second axis gives a second point.
            other_axis = gate_axes[1]
            p_other = bisect(*axes[other_axis][0])
            boundary_pts.append(p_other)
        else:
            # One gate coordinate can be too weak to change vertex labels.
            # Flip each other coordinate right next to a known transition.
            eps = .001
            found=[]
            for j in range(5):
                if j==main_axis:
                    continue
                toggle=[list(p_main),list(p_main)]
                for offset,ptest in zip((-eps,eps),toggle):
                    ptest[main_axis]=min(1.0,max(0.0,ptest[main_axis]+offset))
                    ptest[j]=1.0-main_edge[0][j]
                refs=[list(p_main),list(p_main)]
                for offset,ptest in zip((-eps,eps),refs):
                    ptest[main_axis]=min(1.0,max(0.0,ptest[main_axis]+offset))
                differences=sum(label(a)!=label(b) for a,b in zip(toggle,refs))
                if differences:
                    found.append(j)
            if found:
                other_axis=found[0]
                gate_axes=[main_axis,other_axis]
                # A parallel crossing line with the other variable toggled
                # directly measures the secondary gate coefficient.
                p0=list(main_edge[0]); p1=list(main_edge[1])
                p0[other_axis]=1.0-p0[other_axis]
                p1[other_axis]=1.0-p1[other_axis]
                if label(p0)!=label(p1):
                    boundary_pts.append(bisect(p0,p1))
                else:
                    # Interpolate that variable until a crossing exists.
                    start=main_edge[0][other_axis]
                    end=1.0-start
                    for alpha in (.5,.25,.75,.1,.9):
                        v=start+(end-start)*alpha
                        p0=list(main_edge[0]);p1=list(main_edge[1]);
                        p0[other_axis]=v;p1[other_axis]=v
                        if label(p0)!=label(p1):
                            boundary_pts.append(bisect(p0,p1))
                            break
                    if len(boundary_pts)<2:
                        # Sweep secondary while primary is placed on a level
                        # close to the known transition.
                        for shift in (-.20,.20,-.35,.35):
                            k=min(1.0,max(0.0,p_main[main_axis]+shift))
                            p0=list(p_main);p1=list(p_main)
                            p0[main_axis]=p1[main_axis]=k
                            p0[other_axis]=0.;p1[other_axis]=1.
                            if label(p0)!=label(p1):
                                boundary_pts.append(bisect(p0,p1))
                                break
            else:
                # Last-resort detection: use conditional shifts of the
                # switching location at a larger set of backgrounds.
                gate_axes=[main_axis]
                for j in range(5):
                    if j==main_axis:
                        continue
                    p0=list(main_edge[0]);p1=list(main_edge[1]);
                    p0[j]=p1[j]=1.0-p0[j]
                    if label(p0)!=label(p1):
                        qpt=bisect(p0,p1,8)
                        if abs(qpt[main_axis]-p_main[main_axis])>.004:
                            gate_axes=[main_axis,j]
                            boundary_pts.append(qpt)
                            break

    if len(gate_axes)>2:
        gate_axes=gate_axes[:2]

    normal = np.zeros(5)
    threshold = 0.0
    if len(gate_axes)==2 and len(boundary_pts)>=2:
        i,j=gate_axes
        a,b=boundary_pts[:2]
        dp=float(a[i]-b[i]); dq=float(a[j]-b[j])
        if abs(dq)>1e-5:
            normal[i]=1.0
            normal[j]=-dp/dq
        elif abs(dp)>1e-5:
            normal[j]=1.0
            normal[i]=-dq/dp
        else:
            normal[i]=normal[j]=1.0
        threshold=float(normal@np.array(a))
        # The "on" side is identified by the positive jump residual.
        known_on=next((p for p in corners if corner_labels[p]), None)
        known_off=next((p for p in corners if not corner_labels[p]), None)
        if known_on is not None and known_off is not None:
            if float(normal@np.array(known_on)) < float(normal@np.array(known_off)):
                normal=-normal; threshold=-threshold
        # Obtain additional boundary intersections to reduce rounding error
        # and make the fitted orientation stable even at shallow slopes.
        jlevels=[0.0,.25,.5,.75,1.0]
        for v in jlevels:
            if oracle.remaining<24 or len(boundary_pts)>=5:
                break
            if abs(normal[i])<1e-8:
                continue
            xi=(threshold-normal[j]*v)/normal[i]
            if not .03 < xi < .97:
                continue
            pp=list(boundary_pts[0]);pp[j]=v
            p0=pp.copy();p1=pp.copy();p0[i]=0.;p1[i]=1.
            if label(p0)==label(p1):
                continue
            new=bisect(p0,p1,11)
            if min(abs(new[j]-old[j]) for old in boundary_pts)<.1:
                continue
            boundary_pts.append(new)
        if len(boundary_pts)>=3:
            pp=np.array([[p[i],p[j]] for p in boundary_pts])
            _, _, vt=np.linalg.svd(pp-pp.mean(axis=0),full_matrices=False)
            vec=vt[-1]
            old=normal[[i,j]]
            if float(vec@old)<0:
                vec=-vec
            normal=np.zeros(5);normal[i],normal[j]=map(float,vec)
            threshold=float(vec@pp.mean(axis=0))
    elif len(gate_axes)==1 and boundary_pts:
        i=gate_axes[0]
        normal[i]=1.0
        threshold=boundary_pts[0][i]
        if corner_labels[tuple([1.0]*5)] == 0:
            normal=-normal;threshold=-threshold
    else:
        # Falls back to a signed main diagonal split, extremely unlikely
        # for a genuine two-variable halfspace.
        normal[0]=normal[1]=1.0
        threshold=1.5

    pair_scores=sorted(((abs(float(poly[2][i,j])),i,j)
                        for i in range(5) for j in range(i+1,5)),reverse=True)
    _,u,v=pair_scores[0]

    def design_row(x, gate):
        return [1.0,*x,float(x[u]*x[v]),float(gate)]

    xy=list(data.items())
    X=np.array([design_row(x, label(x)) for x,y in xy])
    Y=np.array([y for x,y in xy])
    theta=np.linalg.lstsq(X,Y,rcond=None)[0]
    # Get a stable estimate even if no data are collected on an interior side.
    if abs(float(theta[-1]))<.02:
        theta[-1]=max(.0,jump0)

    def predict(x):
        val=float(theta[0]+sum(theta[i+1]*x[i] for i in range(5))+
                  theta[6]*x[u]*x[v])
        if sum(normal[i]*x[i] for i in range(5)) >= threshold:
            val+=float(theta[7])
        return val

    # A few direct measurements of held-out points close to the boundary
    # guard against the inevitable 1e-3 tolerance in bisection geometry.
    pw=list(points)
    denom=max(1e-12,float(np.linalg.norm(normal)))
    ranked=sorted(range(len(pw)),key=lambda k:abs(float(normal@np.asarray(pw[k]))-threshold)/denom)
    for k in ranked:
        if oracle.remaining<=8:
            break
        if abs(float(normal@np.asarray(pw[k]))-threshold)/denom > .045:
            break
        ask(pw[k], 'near-boundary held-out')

    preds=[float(ask(p,'cached challenge') if key(p) in data else predict(p)) for p in points]
    gate_names=[VARS[i] for i in sorted(set(gate_axes))]
    relevant={u,v,*gate_axes}
    for i in range(5):
        if abs(float(theta[i+1]))>.015:
            relevant.add(i)
    relevant_names=[VARS[i] for i in range(5) if i in relevant]
    inert_names=[VARS[i] for i in range(5) if i not in relevant]
    expr=[f'({theta[0]:.12g})']
    expr.extend(f'({theta[i+1]:.12g})*x{i+1}' for i in range(5))
    expr.append(f'({theta[6]:.12g})*x{u+1}*x{v+1}')
    cond=' + '.join(f'({normal[i]:.12g})*x{i+1}' for i in range(5) if abs(normal[i])>1e-10)
    expr.append(f'({theta[7]:.12g} if ({cond}) >= ({threshold:.12g}) else 0.0)')
    return {
        'relevant_variables': relevant_names,
        'irrelevant_variables': inert_names,
        'interactions': [{'variables':[VARS[u],VARS[v]],'type':'product',
                          'coefficient':float(theta[6])}],
        'discontinuities':[{'gate_variables':gate_names,
                            'boundary':{'form':'linear',
                                        'coefficients':{VARS[i]:float(normal[i]) for i in range(5)},
                                        'threshold':float(threshold),
                                        'active_when':'greater_equal'},
                            'jump':float(theta[7])}],
        'model_expression':' + '.join(expr),
        'predictions':[round(y,6) for y in preds],
        'notes':f'Adaptive local factorial plus global corners; sparse interaction, '
                f'edge bisection and halfspace inference; {oracle.spent}/150 queries.'
    }
