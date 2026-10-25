"""Aggregate the Foursquare TSMC2014 check-ins (Yang et al., IEEE TSMC 2015) into per-venue tables.

Download https://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip, extract it into
datasets/foursquare/raw/, then run `python datasets/prepare_foursquare.py` from the RPP folder.
Writes datasets/foursquare/venues-{NYC,TKY}.csv with one row per venue and monthly check-in counts.
"""
import csv
import datetime
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foursquare')
RAW = os.path.join(ROOT, 'raw', 'dataset_tsmc2014')

# venues that are not places to visit on a tour (private, work, transit, infrastructure)
EXCLUDED_CATEGORIES = {
    'Home (private)', 'Residential Building (Apartment / Condo)', 'Office', 'Building',
    'Government Building', 'College Academic Building', 'Medical Center', 'School',
    'Train Station', 'Subway', 'Bus Station', 'Light Rail', 'Airport', 'Road', 'Neighborhood',
    'Bridge', 'Gas Station / Garage', 'Bank', 'Post Office', 'Convenience Store',
    'Drugstore / Pharmacy', 'Salon / Barbershop', 'Gym / Fitness Center', 'Deli / Bodega',
}
MIN_CHECKINS = 10
MONTHS = ['2012-04', '2012-05', '2012-06', '2012-07', '2012-08', '2012-09',
          '2012-10', '2012-11', '2012-12', '2013-01', '2013-02']


def prepare(city):
    lats, lons = defaultdict(list), defaultdict(list)
    category, users = {}, defaultdict(set)
    monthly = defaultdict(lambda: np.zeros(len(MONTHS), dtype=int))
    month_index = {m: i for i, m in enumerate(MONTHS)}

    with open(os.path.join(RAW, f'dataset_TSMC2014_{city}.txt'), encoding='latin-1') as f:
        for user, venue, _, cat, lat, lon, _, utc in csv.reader(f, delimiter='\t'):
            if cat in EXCLUDED_CATEGORIES:
                continue
            month = datetime.datetime.strptime(utc, '%a %b %d %H:%M:%S +0000 %Y').strftime('%Y-%m')
            category[venue] = cat
            lats[venue].append(float(lat))
            lons[venue].append(float(lon))
            users[venue].add(user)
            monthly[venue][month_index[month]] += 1

    out = os.path.join(ROOT, f'venues-{city}.csv')
    kept = 0
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['venueID', 'category', 'lat', 'lon', 'checkins', 'users'] + MONTHS)
        for venue, counts in monthly.items():
            if counts.sum() < MIN_CHECKINS:
                continue
            writer.writerow([venue, category[venue], f'{np.median(lats[venue]):.6f}',
                             f'{np.median(lons[venue]):.6f}', counts.sum(), len(users[venue])]
                            + counts.tolist())
            kept += 1
    print(f'{city}: {kept} venues with >= {MIN_CHECKINS} check-ins -> {out}')


if __name__ == '__main__':
    for city in ['NYC', 'TKY']:
        prepare(city)
