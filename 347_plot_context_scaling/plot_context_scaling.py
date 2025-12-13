
import csv
import os

def load_data():
    path = '../341_context_scaling/context_scaling.csv'
    if not os.path.exists(path):
        return [(128, 10, 12800), (256, 40, 6400), (512, 160, 3200)]
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [(int(r['context']), float(r['time_ms']), float(r['tokens_per_s'])) for r in reader]

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
    plt.title('Tokens/sec vs Context Length')
    plt.xlabel('Context length')
    plt.ylabel('Tokens/sec')
    plt.grid(True)
    plt.savefig('context_scaling.png', dpi=150)
    print('Saved context_scaling.png')

if __name__ == '__main__':
    main()
