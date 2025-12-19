
        import csv
        import os

        def load_csv(path, key_col):
            if not os.path.exists(path):
                return []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)

        def main():
            batch = load_csv('../340_batch_scaling/batch_scaling.csv', 'batch')
            context = load_csv('../341_context_scaling/context_scaling.csv', 'context')
            precision = load_csv('../345_precision_compare/precision_compare.csv', 'precision')

            lines = []
            lines.append('# Consolidated Performance Report')
            lines.append('')
            lines.append('## Batch Scaling')
            if batch:
                for row in batch:
                    lines.append(f"- batch {row['batch']}: {row['avg_ms']} ms, {row['throughput_imgs_s']} imgs/s")
            else:
                lines.append('- batch scaling data not found')

            lines.append('')
            lines.append('## Context Scaling')
            if context:
                for row in context:
                    lines.append(f"- context {row['context']}: {row['time_ms']} ms, {row['tokens_per_s']} tokens/s")
            else:
                lines.append('- context scaling data not found')

            lines.append('')
            lines.append('## Precision Comparison')
            if precision:
                for row in precision:
                    lines.append(f"- {row['precision']}: {row['avg_ms']} ms")
            else:
                lines.append('- precision data not found')

            with open('consolidated_report.md', 'w', encoding='utf-8') as f:
                f.write('
'.join(lines) + '
')
            print('Wrote consolidated_report.md')

        if __name__ == '__main__':
            main()
