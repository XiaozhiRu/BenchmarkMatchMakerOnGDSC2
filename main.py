import os
import numpy as np
import pandas as pd
import tensorflow as tf
import MatchMaker
import performance_metrics
import argparse

print("---------- Main Script Started! ----------")

print("Parsing Arguments ...")
# --------------- Parse MatchMaker arguments --------------- #

parser = argparse.ArgumentParser(description='REQUEST REQUIRED PARAMETERS OF MatchMaker')

parser.add_argument('--Comb', default='./train_set/DrugCombinationData.tsv', # Edit: current directory
                    help="Name of the test drug combination data")

parser.add_argument('--comb-data-name', default='test_comb_data.csv', # Edit: current directory
                    help="Name of the test drug combination data")

parser.add_argument('--cell_line-gex', default='./train_set/cell_line_gex.csv',
                    help="Name of the cell line gene expression data")

parser.add_argument('--drug1-chemicals', default='./train_set/drug1_chem.csv',
                    help="Name of the chemical features data for drug 1")

parser.add_argument('--drug2-chemicals', default='./train_set/drug2_chem.csv',
                    help="Name of the chemical features data for drug 2")

parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

parser.add_argument('--train-test-mode', default=0, type = int,
                    help="Test of train mode (0: test, 1: train)")

parser.add_argument('--train-ind', default='./train_set/train_inds.txt',
                    help="Data indices that will be used for training")

parser.add_argument('--test_cell_line', default='test_cell_line.csv',
                    help="Test cell line data.")

parser.add_argument('--test-drug1-chemicals', default='test_chem1.csv',
                    help="Name of the chemical features data for drug 1")

parser.add_argument('--test-drug2-chemicals', default='test_chem2.csv',
                    help="Name of the chemical features data for drug 2")

parser.add_argument('--arch', default='architecture.txt',
                    help="Architecute file to construct MatchMaker layers")

parser.add_argument('--gpu-support', default=True,
                    help='Use GPU support or not')

parser.add_argument('--saved-model-name', default="matchmaker_saved.h5",
                    help='Model name to save weights')
args = parser.parse_args()
# ---------------------------------------------------------- #
num_cores = 8
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
GPU = True
if args.gpu_support:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 2
    num_GPU = 0

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )

tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

print("Loading the train data to get the parameters ...")
# load train data and get the parameters.
chem1, chem2, cell_line, train_synergies = MatchMaker.data_loader(args.drug1_chemicals, args.drug2_chemicals,
                                                args.cell_line_gex, args.Comb)
drug1, drug2, test_cell, test_synergies = MatchMaker.test_loader(args.test_drug1_chemicals, args.test_drug2_chemicals,
                                                            args.test_cell_line, args.comb_data_name)

print("Normalizing and Splitting Data ...")
# normalize and split data into train, validation and test
norm = 'tanh_norm'
train_data, test_data = MatchMaker.prepare_data(chem1, chem2, cell_line, train_synergies, norm,
                                    args.train_ind, drug1, drug2, test_cell, test_synergies)


print("Calculating Weights ...")
# calculate weights for weighted MSE loss
min_s = np.amin(train_data['y'])
loss_weight = np.log(train_data['y'] - min_s + np.e)

# load architecture file
architecture = pd.read_csv('architecture.txt')

# prepare layers of the model and the model name
layers = {}
layers['DSN_1'] = architecture['DSN_1'][0] # layers of Drug Synergy Network 1
layers['DSN_2'] = architecture['DSN_2'][0] # layers of Drug Synergy Network 2
layers['SPN'] = architecture['SPN'][0] # layers of Synergy Prediction Network
modelName = args.saved_model_name # name of the model to save the weights

# define constant parameters of MatchMaker
l_rate = 0.0001
inDrop = 0.2
drop = 0.5
max_epoch = 1000
batch_size = 128
earlyStop_patience = 100

model = MatchMaker.generate_network(train_data, layers, inDrop, drop)

# if (args.train_test_mode == 1):
#     # if we are in training mode
#    model = MatchMaker.trainer(model, l_rate, train_data, val_data, max_epoch, batch_size,
#                                earlyStop_patience, modelName,loss_weight)

# load the best model
model.load_weights(modelName)

# for layer in model.layers:
#     weights = layer.get_weights()
#     print(f"Layer {layer.name} weights: {weights}")

print("Predicting ...")
# predict in Drug1, Drug2 order
pred1 = MatchMaker.predict(model, [test_data['drug1'],test_data['drug2']])
np.savetxt("pred1.txt", np.asarray(pred1), delimiter=",")
np.savetxt("y_test.txt", np.asarray(test_data['y']), delimiter=",")
print("Predict in Drug1, Drug2 order...")
mse_value = performance_metrics.mse(test_data['y'], pred1)
spearman_value = performance_metrics.spearman(test_data['y'], pred1)
pearson_value = performance_metrics.pearson(test_data['y'], pred1)


# predict in Drug2, Drug1 order
pred2 = MatchMaker.predict(model, [test_data['drug2'],test_data['drug1']])
# take the mean for final prediction
pred = (pred1 + pred2) / 2

mse_value = performance_metrics.mse(test_data['y'], pred)
spearman_value = performance_metrics.spearman(test_data['y'], pred)
pearson_value = performance_metrics.pearson(test_data['y'], pred)


# save the predictions to a dataframe for analysis (include the drug_row, drug_col, cell_line_name, y, pred)
# print("Comparing y and pred ...")

# print("Creating Dataframe: | drug_row | drug_col | cell_line_name | synergy_loewe |")
# def compare(y, pred):
    # comparison = pd.DataFrame({'y': y, 'pred': pred})
    # comparison.to_csv('comparison.csv', index=False)
    # print("Comparison of y and pred saved to comparison.csv")

# compare(test_data['y'], pred)



print(test_data['drug1'].shape)
print(test_data['drug1'].shape)
print(test_data['drug1'].shape)

print("---------- Main.py is done! ----------")
