import ast
import matplotlib.pyplot as plt

DATA_PATH = {'train': '/media/ubuntu/Data/Data/MNRE/mnre/data/mnre/txt/ours_train.txt',
             'dev': '/media/ubuntu/Data/Data/MNRE/mnre/data/mnre/txt/ours_val.txt',
             'test': '/media/ubuntu/Data/Data/MNRE/mnre/data/mnre/txt/ours_test.txt',
}

modes = ['train', 'dev', 'test']

for mode in modes:
    load_file = DATA_PATH[mode]
    print("Loading data from {}".format(load_file))
    with open(load_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        relations = {}
        for i, line in enumerate(lines):
            line = ast.literal_eval(line)
            relation = line['relation']
            if relation not in relations.keys():
                relations[relation] = 1
            else:
                relations[relation] += 1

        # Extract keys and values for plotting
        labels = list(relations.keys())
        values = list(relations.values())

        # Create a horizontal bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(labels, values)
        plt.ylabel('Relations')
        plt.xlabel('Counts')
        plt.title(f'{mode} set Relation Counts')
        plt.tight_layout()  # Adjust layout to make room for labels

        plt.show()

