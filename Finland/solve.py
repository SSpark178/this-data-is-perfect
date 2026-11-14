"""Reconstruct the undirected Finnish route graph from the organizer CSV."""
import pandas as pd
import unicodedata
from collections import Counter
from pathlib import Path

ALIASES = {
    'abo':'turku','alavo':'alavus','birkala':'pirkkala','bjorneborg':'pori','borga':'porvoo','brahestad':'raahe',
    'esbo':'espoo','fredrikshamn':'hamina','grankulla':'kauniainen','hango':'hanko','helsingfors':'helsinki',
    'hogfors':'karkkila','hyvinge':'hyvinkaa','idensalmi':'iisalmi','ikalis':'ikaalinen','ilmola':'ilmajoki',
    'inga':'inkoo','jakobstad':'pietarsaari','kajana':'kajaani','karleby':'kokkola','kasko':'kaskinen',
    'kervo':'kerava','keuru':'keuruu','kimitoon':'kemionsaari','korsholm':'mustasaari',
    'kristinestad':'kristiinankaupunki','kronoby':'kruunupyy','kumo':'kokemaki','kyrkslatt':'kirkkonummi',
    'lahtis':'lahti','lappo':'lapua','larsmo':'luoto','lojo':'lohja','lovisa':'loviisa','lundo':'lieto','malax':'maalahti',
    'nadendal':'naantali','narpes':'narpio','nykarleby':'uusikaarlepyy','nystad':'uusikaupunki','pargas':'parainen',
    'raumo':'rauma','raseborg':'raasepori','reso':'raisio','s:t karins':'kaarina','s:t michel':'mikkeli','sibbo':'sipoo',
    'sjundea':'siuntio','tammerfors':'tampere','tavastehus':'hameenlinna','tornea':'tornio','traskanda':'jarvenpaa',
    'tusby':'tuusula','uleaborg':'oulu','ulvsby':'ulvila','vanda':'vantaa','vasa':'vaasa','vichtis':'vihti',
    'villmanstrand':'lappeenranta','virmo':'mynamaki','vora':'voyri'
}

def canonical(name):
    base = name.strip().rstrip('.').casefold()
    base = ''.join(ch for ch in unicodedata.normalize('NFKD', base) if not unicodedata.combining(ch))
    return ALIASES.get(base, base)

def main():
    path = Path(__file__).with_name('routes.csv')
    routes = pd.read_csv(path)
    sources = routes['from_town'].map(canonical)
    targets = routes['to_town'].map(canonical)
    cities = set(sources) | set(targets)
    edges = {tuple(sorted((a, b))) for a, b in zip(sources, targets) if a != b}
    degrees = Counter(city for edge in edges for city in edge)
    ranked = degrees.most_common(8)

    print('Cities:', len(cities))
    print('Distinct undirected connections:', len(edges))
    for rank, (city, degree) in enumerate(ranked, 1):
        print(f'{rank}. {city.title()}: {degree}')
    flag = 'SIGMOID_' + ''.join(city[0].upper() for city, degree in ranked[:7])
    print('Flag:', flag)

    assert len(cities) == 116, 'City normalization did not match the official validation'
    assert len(edges) == 448, 'Edge normalization did not match the official validation'

if __name__ == '__main__':
    main()
