from old.database import get_local_database


def find_all_to_indexes(edges, to):
    for (i, edge) in enumerate(edges):
        if edge['to'] == to:
            yield i


def find_leading_information(edges, start_index, timeframe):
    start_date = edges[start_index]['date']
    for i in reversed(range(0, start_index)):
        edge = edges[i]
        time_delta = abs(edge['date'] - start_date)
        if time_delta.days <= timeframe:
            yield edge


def find_all_leading_information(edges, end, timeframe):
    matching_to_indexes = list(find_all_to_indexes(edges, end))
    for index in matching_to_indexes:
        leading_information = list(find_leading_information(edges, index, timeframe))
        leading_information.sort(key=lambda i: i['date'])
        yield leading_information


if __name__ == '__main__':

    TIMEFRAME = 30
    NUMBER_OF_MOST_KNOWN_NODES_TAKIN_INTO_ACCOUNT = 1

    db = get_local_database()
    mpn = db['most_popular_nodes']
    most_popular_node = mpn.find(sort=[('knownByCount', -1)], limit=NUMBER_OF_MOST_KNOWN_NODES_TAKIN_INTO_ACCOUNT)[0]
    end = most_popular_node['_id']
    starts = most_popular_node['knownBy']

    ife = db['information_flow_edges']
    li = db[f'leading_information_{TIMEFRAME}days']
    for (i_start, start) in enumerate(starts):
        print(f'Processing {i_start} / {len(starts)}')
        edges = list(ife.find(sort=[('date', 1)], filter={'from': start}))
        matching_to_indexes = list(find_all_to_indexes(edges, end))
        for index in matching_to_indexes:
            leading_information = list(find_leading_information(edges, index, TIMEFRAME))
            _id = f"{start} -> {end} / {edges[index]['_id']}"
            doc = {
                '_id': _id,
                'from': start,
                'to': end,
                'leading_to': edges[index],
                'leading_information': leading_information
            }
            li.replace_one({'_id': _id}, doc, True)
            # print(f'Saved (timeframe {TIMEFRAME}days): {doc}')

    print('Finished')
