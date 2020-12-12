import json

label_map = {}
num_map = {}
patient_map = {}
i = 0
with open('data/GSE13159.info', 'r') as f:
    f.readline()
    for line in f:
        leukemia = line.split('  ')[-1].strip()
        if leukemia not in label_map:
            label_map[leukemia] = i
            num_map[i] = leukemia
            i += 1
        patient = line.split()[0]
        patient_map[patient] = label_map[leukemia]

with open('mappings/label_to_num.json', 'w') as f:
    json.dump(label_map, f, indent=4)
with open('mappings/num_to_label.json', 'w') as f:
    json.dump(num_map, f, indent=4)
with open('mappings/patient_to_type.json', 'w') as f:
    json.dump(patient_map, f, indent=4)

i = 0
feature_map = {}
description_map = {}
with open('data/GSE13159.U133Plus2_EntrezCDF.MAS5.log2.pcl', 'r') as f:
    f.readline()
    for line in f:
        name, description = line.split('\t')[0:2]
        feature_map[i] = name
        description_map[name] = description
        i += 1
with open('mappings/num_to_feature.json', 'w') as f:
    json.dump(feature_map, f, indent=4)
with open('mappings/feature_to_desc.json', 'w') as f:
    json.dump(description_map, f, indent=4)
    