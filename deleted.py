# necessary libraries
import random
from collections import Counter
import pandas
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')

personalities = ['F', 'S']  # F for flexible, S for stubborn
sentence_type = ['reversible', 'irreversible']  # two kinds of sentences
sentence_weights = [7, 3]  # humans use more reversible sentences than irreversible ones, so, weights should be different

basic_orders = ['OSV', 'OVS', 'SOV', 'SVO', 'VOS', 'VSO']  # possible 6 orders for basic sentences, maybe "no order/other" can be added

# for children making process
start = 1
stop = 3
MAX_GROUP_SIZE = 7

# starting bias weights for corresponding word order in basic_orders
RANDOM_IRREV_BIAS = random.sample(range(0, 100), 6)
RANDOM_REV_BIAS = random.sample(range(0, 100), 6)

k = 3  # factor
n = 25  # number of people in one starting group

error = 10  # add to the weights
personality_weight = [5, 5]  # equally
coeff_stubborn = 10
coeff_flexible = 1
error_or_pressure_rate = 0.00001

uniformIRREV = []
uniformREV = []

#REV_BIAS = [1, 1, 1, 1, 1, 5]
#REV_BIAS = [i * coeff for i in REV_BIAS]

REV_BIAS = [1, 1, 1, 1, 1, 5]
#REV_BIAS = [i * coeff for i in REV_BIAS]
IRREV_BIAS = [1, 5, 1, 1, 1, 1]
#IRREV_BIAS = [i * coeff for i in IRREV_BIAS]
#starting_irrev_bias = uniformIRREV
#starting_rev_bias = uniformREV

tendency = [3, 8, 410, 350, 20, 70]  # world's current distribution
TEND_COM = random.choices(basic_orders, weights=tendency, k=1)
TEND_REV = 'SVO'  # what we know
TEND_IRREV = 'SOV'  # what we know

def add_rev_weights(self, word_order):
    self.rev_weights = self.list_summation(self.rev_weights, self.new_weight_rev(word_order))


def add_irrev_weights(self, word_order):
    self.irrev_weights = self.list_summation(self.irrev_weights, self.new_weight_irrev(word_order))


def new_weight_rev(self, order):  # for updating the stubborn agent's word order weights
    weight = []
    for i in basic_orders:
        if order == i or i == TEND_COM[0]:
            weight.append(error_or_pressure_rate)  # if the used/given order is what it is, than add 1 to the weight
        else:
            weight.append(0)  # if it is not, add 0 (nothing)
    return weight


def new_weight_irrev(self, order):  # for updating the stubborn agent's word order weights
    weight = []
    for i in basic_orders:
        if order == i or i == TEND_COM[0]:
            weight.append(error_or_pressure_rate)  # if the used/given order is what it is, than add 1 to the weight
        else:
            weight.append(0)  # if it is not, add 0 (nothing)
    return weight


def new_weight_with_error_irrev(self, order):  # for updating the flexible agent's word order weights
    weight = []
    for i in basic_orders:
        if order == i or i == TEND_COM[0]:
            weight.append(error_or_pressure_rate)  # add 1 to the corresponding word order's place
        else:
            weight.append(random.uniform(-1, 1))  # there should be a error
    return weight


def new_weight_with_error_rev(self, order):  # for updating the flexible agent's word order weights
    weight = []
    for i in basic_orders:
        if order == i or i == TEND_COM[0]:
            weight.append(error_or_pressure_rate)  # add 1 to the corresponding word order's place
        else:
            weight.append(random.uniform(-1, 1))  # there should be a error
    return weight