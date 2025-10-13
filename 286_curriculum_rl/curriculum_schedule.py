import csv


def main():
    with open("curriculum_schedule.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "difficulty"])
        for epoch in range(10):
            difficulty = 0.1 + epoch * 0.1
            writer.writerow([epoch, difficulty])
    print("Wrote curriculum_schedule.csv")


if __name__ == "__main__":
    main()
