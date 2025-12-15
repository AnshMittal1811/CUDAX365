
import csv
import os

def load_data():
    path = '../345_precision_compare/precision_compare.csv'
    if not os.path.exists(path):
        return [('fp32', 5.0), ('fp16', 2.5), ('int8', 4.0)]
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = []
        for r in reader:
            try:
                val = float(r['avg_ms'])
            except Exception:
                val = None
            data.append((r['precision'], val))
        return data

def main():
    data = load_data()
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib not available')
        print(data)
        return
    labels = [d[0] for d in data]
    values = [d[1] if d[1] is not None else 0 for d in data]
    plt.bar(labels, values)
    plt.title('Latency by Precision')
    plt.ylabel('Avg ms')
    plt.savefig('precision_compare.png', dpi=150)
    print('Saved precision_compare.png')

if __name__ == '__main__':
    main()
