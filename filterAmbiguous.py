import json, jsonlines
from collections import Counter


def filter5LabeledTrain(inputFile):
    # Open input file and output file
    with jsonlines.open(inputFile) as infile, jsonlines.open('output.jsonl', mode='w') as outfile:
        # Iterate over each line in the input file
        for line in infile:
            # Check if the length of the "annotator_labels" list is 5
            if len(line['annotator_labels']) == 5:
                # Write the line to the output file
                outfile.write(line)

def discardedNoGoldLabel(input_file):
    ambiguous_file = 'ambiguousDiscarded.jsonl'
    nonambiguous_file = 'nonambiguousDiscarded.jsonl'
    
    with jsonlines.open(input_file) as infile, \
         jsonlines.open(ambiguous_file, mode='w') as ambiguous_outfile, \
         jsonlines.open(nonambiguous_file, mode='w') as nonambiguous_outfile:

        for line in infile:
            # Ignore examples with gold_label "-"
            if line['gold_label'] == '-':
                continue
            
            label_counts = Counter(line['annotator_labels'])
            max_count = max(label_counts.values())

            if max_count < 4:
                ambiguous_outfile.write(line)
            else:
                nonambiguous_outfile.write(line)

def addedDoubleGoldLabels_counter(input_file):
    ambiguous = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            annotator_labels = data['annotator_labels']
            gold_label = data['gold_label']
            
            if gold_label == '-':
                label_counts = Counter(annotator_labels)
                most_common = label_counts.most_common(2)
                
                if len(most_common) != 2:
                    continue
                
                label1, count1 = most_common[0]
                label2, count2 = most_common[1]
                
                if count1 != 2 or count2 != 2:
                    continue
                
                for label in [label1, label2]:
                    new_data = data.copy()
                    new_data['gold_label'] = label
                    ambiguous.append(new_data)
                    
            else:
                label_counts = Counter(annotator_labels)
                max_count = max(label_counts.values())
                
                if max_count < 4:
                    ambiguous.append(data)
                    
    with open('ambiguousDouble.jsonl', 'w') as f:
        for data in ambiguous:
            f.write(json.dumps(data) + '\n')

discardedNoGoldLabel('filtered.jsonl')
addedDoubleGoldLabels_counter('filtered.jsonl')