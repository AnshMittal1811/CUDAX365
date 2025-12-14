
import csv
import os

def load_data():
    path = '../340_batch_scaling/batch_scaling.csv'
    if not os.path.exists(path):
        return [(1, 2.0, 500.0), (2, 3.0, 666.0), (4, 4.5, 888.0)]
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [(int(r['batch']), float(r['avg_ms']), float(r['throughput_imgs_s'])) for r in reader]

def main():
    data = load_data()
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib not available')
        print(data)
        return
    x = [d[0] for d in data]
    y = [d[2] for d in data]
    plt.plot(x, y, marker='o')
    plt.title('Throughput vs Batch Size')
    plt.xlabel('Batch size')
    plt.ylabel('Images/sec')
    plt.grid(True)
    plt.savefig('batch_scaling.png', dpi=150)
    print('Saved batch_scaling.png')

if __name__ == '__main__':
    main()
