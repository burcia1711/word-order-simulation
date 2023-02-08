# necessary libraries

import random
from collections import Counter
import pandas

import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('svg', 'pdf')

personalities = ['F', 'S']  # F for flexible, S for stubborn

sentence_type = ['reversible', 'irreversible']  # two kinds of sentences
sentence_weights = [7,
                    3]  # humans use more reversible sentences than irreversible ones, so, weights should be different

basic_orders = ['OSV', 'OVS', 'SOV', 'SVO', 'VOS',
                'VSO']  # possible 6 orders for basic sentences, maybe "no order/other" can be added

# for children making process
start = 1
stop = 3
MAX_GROUP_SIZE = 7
# starting bias weights for corresponding word order in basic_orders
uniformIRREV = [1] * 6
uniformREV = [1] * 6

RANDOM_IRREV_BIAS = random.sample(range(0, 100), 6)
RANDOM_REV_BIAS = random.sample(range(0, 100), 6)

k = 2
n = 25  # a group
error = 10

coeff = 1

REV_BIAS = [1, 1, 1, 5, 1, 1]
REV_BIAS = [i * coeff for i in REV_BIAS]
IRREV_BIAS = [1, 1, 5, 1, 1, 1]
IRREV_BIAS = [i * coeff for i in IRREV_BIAS]


starting_irrev_bias = REV_BIAS
starting_rev_bias = IRREV_BIAS


tendency = [3, 8, 410, 350, 20, 70]
error_or_pressure_rate = 0.00001

TEND_COM = random.choices(basic_orders, weights=tendency, k=1)
TEND_REV = 'SVO'  # what we know
TEND_IRREV = 'SOV'  # what we know


class Agent:

    def __init__(self, g, p, mother=None, father=None):  # First agent has no parent, children will.
        self.generation = g
        self.personality = p
        if mother is None and father is None:  # first generation, starting point
            self.irrev_weights = starting_irrev_bias
            self.rev_weights = starting_rev_bias
        else:  # if it is a child, mother & father should affect
            self.irrev_weights = self.set_irrev_weights(mother, father)
            self.rev_weights = self.set_rev_weights(mother, father)

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

    def new_weight_with_pressure_irrev(self, order):  # some pressures made us eliminate others
        weight = []
        # error_or_pressure_rate = random.uniform(0, 0.01)
        for i in basic_orders:
            if i == order or i == TEND_COM[0]:  # or i == TEND_IRREV:
                weight.append(error_or_pressure_rate*(error**k)) # add 1 to the used word order
            else:
                weight.append(-error_or_pressure_rate*(error**k))  # add -1 to weights of non-used word orders
        return weight

    def new_weight_with_pressure_rev(self, order):  # some pressures made us eliminate others
        weight = []
        # error_or_pressure_rate = random.uniform(0, 0.01)
        for i in basic_orders:
            if i == order or i == TEND_COM[0]:  # or i == TEND_REV:
                weight.append(error_or_pressure_rate*(error**k))  # add 1 to the used word order
            else:
                weight.append(-error_or_pressure_rate*(error**k))  # add -1 to weights of non-used word orders
        return weight

    def list_summation(self, l1, l2):  # adding two lists
        res_lt = [l1[x] + l2[x] for x in range(len(l1))]
        return res_lt

    def list_average(self, l1, l2):  # for averaging mother and father's weights
        res_lt = [(l1[x] + l2[x]) for x in range(len(l1))]
        return res_lt

    def is_stubborn(self):  # check if stubborn
        return self.personality == 'S'

    def set_irrev_weights(self, mother,
                          father):  # calculate average of mother+father weights for irreversible sentences
        return self.list_average(mother.irrev_weights, father.irrev_weights)

    def set_rev_weights(self, mother, father):  # calculate average of mother+father weights for reversible sentences
        return self.list_average(mother.rev_weights, father.rev_weights)

    def add_rev_weights(self, word_order):
        if self.is_stubborn():
            self.rev_weights = self.list_summation(self.rev_weights, self.new_weight_rev(word_order))
        else:
            multiplied_list = [element * 1 for element in self.new_weight_with_error_rev(word_order)]
            self.rev_weights = self.list_summation(self.rev_weights, multiplied_list)

    def add_irrev_weights(self, word_order):
        if self.is_stubborn():
            self.irrev_weights = self.list_summation(self.irrev_weights, self.new_weight_irrev(word_order))
        else:
            multiplied_list = [element * 1 for element in self.new_weight_with_error_irrev(word_order)]
            self.irrev_weights = self.list_summation(self.irrev_weights, multiplied_list)

    def add_rev_weights_with_pressure(self, word_order):
        if self.is_stubborn():
            self.rev_weights = self.list_summation(self.rev_weights, self.new_weight_with_pressure_rev(word_order))
        else:
            multiplied_list = [element * 1 for element in self.new_weight_with_pressure_rev(word_order)]
            self.rev_weights = self.list_summation(self.rev_weights, multiplied_list)

    def add_irrev_weights_with_pressure(self, word_order):
        if self.is_stubborn():
            self.irrev_weights = self.list_summation(self.irrev_weights,
                                                     self.new_weight_with_pressure_irrev(word_order))
        else:
            multiplied_list = [element * 1 for element in self.new_weight_with_pressure_irrev(word_order)]
            self.irrev_weights = self.list_summation(self.irrev_weights, multiplied_list)


# print agents
def print_agent(agent):
    return [agent.generation, agent.personality, agent.irrev_weights, agent.rev_weights]


def make_first_gen_agents(N):  # create N number of agents with different random personalities
    gen = 1
    population = []

    for i in range(N):
        p = random.randint(0, 1)
        agent = Agent(gen, personalities[p])
        population.append(agent)

    # for obj in population:
    # print(obj.rev_weights)

    print(TEND_COM)
    return population


def create_children(mother, father,
                    number_of_children):  # create children of given mother and father, with the given number of children
    children = []
    for i in range(number_of_children):
        p = random.randint(0, 1)
        child = Agent(mother.generation + 1, personalities[p], mother, father)
        children.append(child)
    return children


def pop_random(lst):  # select random pairs from a list
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)


def create_pairs(population):  # select mother+father pairs from a population
    lst = list(range(0, len(population)))
    pairs = []
    while len(lst) > 1:
        rand1 = pop_random(lst)
        rand2 = pop_random(lst)
        pair = rand1, rand2
        pairs.append(pair)
    return pairs


def calculate_average_children_number_per_family(length_pop):  # calculate the average number of each mother+father pair
    average_children = round((length_pop * (random.randrange(start, stop + 1) + 0.3) / length_pop))
    # print(average_children)
    return average_children


def create_generation(prev_generation_pop):
    population_length = len(prev_generation_pop)
    pairs = create_pairs(prev_generation_pop)
    next_gen = []
    for p in pairs:
        children_number = calculate_average_children_number_per_family(population_length)
        next_gen.extend(create_children(prev_generation_pop[p[0]], prev_generation_pop[p[1]], children_number))
    # print(len(next_gen))
    return next_gen


# tt = []
# tt.extend(make_first_gen_agents(15))
# print(tt)
# newly_whole_population = []
# newly_whole_population.extend(tt)
# print(len(newly_whole_population))
# for i in range(10):
#  current_prev = tt[:]
#  #print(f"current prev {current_prev}")
#  tt.extend(create_generation(current_prev))
#  newly_whole_population.extend(tt)
#  print(len(newly_whole_population))


def plot_freq_list(lst, ttle):
    count = Counter(sorted(lst))
    df = pandas.DataFrame.from_dict(count, orient='index')
    df.plot(kind='bar', color="violet")
    plt.title(ttle)
    plt.show()


def generate_word_order_list(order_list, weight, n):
    return random.choices(order_list, weights=weight, k=n)


# IRREV_WORD_ORDER_LIST = []
# REV_WORD_ORDER_LIST = []

# IRREV_WORD_ORDER_LIST = generate_word_order_list(basic_orders, starting_irrev_bias, 100)
# REV_WORD_ORDER_LIST = generate_word_order_list(basic_orders, starting_rev_bias, 100)


def make_utterance(n):
    return random.choices(sentence_type, weights=sentence_weights, k=n)


# utterance_list =[]
# utterance_list = make_utterance(500)

# plot_freq_list(utterance_list, "sentences")
# plot_freq_list(IRREV_WORD_ORDER_LIST, "irrev")
# plot_freq_list(REV_WORD_ORDER_LIST, "rev")


def select_two_random_persons(population):
    people_selected = []

    agent1_index = random.randint(0, len(population) - 1)
    agent2_index = random.randint(0, len(population) - 1)

    while agent1_index == agent2_index:
        agent2_index = random.randint(0, len(population))

    print(agent1_index)
    print(agent2_index)

    people_selected.append(population[agent1_index])
    people_selected.append(population[agent2_index])
    # print(people_selected, sentence_list)

    return people_selected


def select_n_random_persons(n_people, population):
    people_indices = random.sample(range(0, len(population)), n_people)
    return people_indices


# two people communicate with n sentences
def two_people_communicate(n,
                           population):  # n is the number of sentences for the communication, population is the given population
    sentence_list = make_utterance(n)  # create n sentence type list (rev or irrev)
    selected_people = select_two_random_persons(population)  # select two random people in the population

    for i in range(n):
        speaker_index = random.randint(0, 1)
        listener_index = int(not speaker_index)
        # print(speaker_index, listener_index)

        if sentence_list[i] == 'irreversible':
            spoken_word_order = generate_word_order_list(basic_orders, selected_people[speaker_index].irrev_weights,
                                                         1)  # generate a word order for given sentence
            # print(spoken_word_order)
            # update listener
            selected_people[listener_index].add_irrev_weights(spoken_word_order[0])

        else:
            spoken_word_order = generate_word_order_list(basic_orders, selected_people[speaker_index].rev_weights,
                                                         1)  # generate a word order for given sentence
            # print(spoken_word_order)
            # update listener
            selected_people[listener_index].add_rev_weights(spoken_word_order[0])


# n people communicate with n sentences
def n_people_communicate(n_people, n_sent,
                         population):  # n_people is the number of people, n_sent is the # of sentences for the communication, population is the given population
    sentence_list = make_utterance(n_sent)  # create n_sent sentence type list (rev or irrev)
    selected_people = select_n_random_persons(n_people, population)  # select n_people random people in the population

    for i in range(n_sent):
        speaker = random.choice(selected_people)
        # print(speaker)

        if sentence_list[i] == 'irreversible':
            spoken_word_order = generate_word_order_list(basic_orders, population[speaker].irrev_weights,
                                                         1)  # generate a word order for given sentence
            # print(f'irrev: {spoken_word_order}')
            # update listeners
            for l in selected_people:
                if l != speaker:
                    population[l].add_irrev_weights(spoken_word_order[0])

        else:
            spoken_word_order = generate_word_order_list(basic_orders, population[speaker].rev_weights,
                                                         1)  # generate a word order for given sentence
            # print(f'rev: {spoken_word_order}')
            # update listeners
            for l in selected_people:
                if l != speaker:
                    population[l].add_rev_weights(spoken_word_order[0])


# Population communication
# n_group: n different groups, n_people: from 2 to n_people generate random numbers of people in each group, n_sent: n sentences spoken in each group, population is the given population
def n_groups_communicate(n_group, n_people, n_sent, population):
    number_of_peoples_in_groups = []
    for g in range(n_group):
        number_of_peoples_in_groups.append(random.randint(2, min(len(population), n_people)))
    # print(number_of_peoples_in_groups)

    for groups in number_of_peoples_in_groups:
        n_people_communicate(groups, n_sent, population)


def random_groups_communication(population, number_of_groups, n_sent):
    # Shuffle list of students
    random.shuffle(population)

    # Create groups
    all_groups = []
    for index in range(number_of_groups):
        group = population[index::number_of_groups]
        all_groups.append(group)

    # Format and display groups
    for index, group in enumerate(all_groups):
        n_people_communicate(len(group), n_sent, group)


# n_groups_communicate(100, 20, 500, newly_whole_population)

# def one_to_many_network_communication(n_people, n_sent, population):


def group_communication_all_population_speaks(n_sent, population):
    groups = create_groups_from_all_population_members(population)
    for group in groups:
        group_communication_all_participants_speak(group, n_sent, population)


def create_groups_from_all_population_members(population):
    groups = []
    population_indices = list(range(0, len(population), 1))
    while len(population_indices) != 0:
        if len(population_indices) <= 3:
            sample = population_indices
        else:
            sample = random.sample(population_indices, random.randint(2, min(len(population_indices), MAX_GROUP_SIZE)))
        groups.append(sample)
        population_indices = list(filter(lambda x: x not in sample, population_indices))

    return groups


def group_communication_all_participants_speak(group_members_indices, n_sent_each, population):
    for speaker in group_members_indices:
        for listener in group_members_indices:
            if speaker != listener:
                sentence_list = make_utterance(n_sent_each)
                for sent in sentence_list:
                    if sent == 'irreversible':
                        spoken_word_order = generate_word_order_list(basic_orders, population[speaker].irrev_weights,
                                                                     1)  # generate a word order for given sentence
                        # update listeners
                        population[listener].add_irrev_weights_with_pressure(spoken_word_order[0])
                        population[speaker].add_irrev_weights_with_pressure(spoken_word_order[0])

                    else:
                        spoken_word_order = generate_word_order_list(basic_orders, population[speaker].rev_weights,
                                                                     1)  # generate a word order for given sentence
                        # update listeners

                        population[listener].add_rev_weights_with_pressure(spoken_word_order[0])
                        population[speaker].add_rev_weights_with_pressure(spoken_word_order[0])


def population_final_word_orders(population, ttle):
    IRREV_LIST = []
    REV_LIST = []

    for people in population:
        IRREV_LIST.extend(generate_word_order_list(basic_orders, people.irrev_weights, 100))
        # print(people.irrev_weights)
        REV_LIST.extend(generate_word_order_list(basic_orders, people.rev_weights, 100))
        # print(people.rev_weights)
        # plt.hist(IRREV_LIST)
        # plt.hist(REV_LIST)

    plot_freq_list(IRREV_LIST, ttle + " irrev")
    plot_freq_list(REV_LIST, ttle + " rev")


def population_personality(population, ttle):
    personality_list = []

    for people in population:
        personality_list.extend(people.personality)

    plot_freq_list(personality_list, ttle)


def pos_generation_range(n_gen, n_range):
    possible_ranges = []
    for i in range(n_gen - n_range + 2):
        l = []
        for j in range(n_range):
            l.append(i + j)
        possible_ranges.append(l)

    return possible_ranges


def select_agents_of_given_gen_range(population, rnge):
    community_index = []
    for index in range(len(population)):
        if population[index].generation in rnge:
            community_index.append(index)
    return community_index


def main_simulation():
    TOTAL_POP = []
    population_first_gen = make_first_gen_agents(k*n)  # create first community with 20 agents
    population_final_word_orders(population_first_gen, "population first word orders")

    ############# COMMUNICATIONS #############
    # n_groups_communicate(100, 50, 10000, population_first_gen)  # n_group: group number
    # n_people: max people in a group,
    # n_sent: number of sentences to speak,
    # population: current population to communicate
    group_communication_all_population_speaks(1000, population_first_gen)
    TOTAL_POP.extend(population_first_gen)

    ############# CHILDREN #############
    last_gen = population_first_gen

    for i in range(10):  # create 10 generations
        new_gen = create_generation(last_gen)  # create a new generation from last generation
        last_gen = new_gen  # make new generation last generation
        TOTAL_POP.extend(new_gen)  # extend total population with last generation
        TOTAL_POP = list(
            filter(lambda person: person.generation >= i - 4, TOTAL_POP))  # filter out except last 4 generations
        # n_groups_communicate(100, 50, 1000, TOTAL_POP) #communicate current living population
        group_communication_all_population_speaks(5000, TOTAL_POP)
        # population_final_word_orders(TOTAL_POP, "population final word orders") #print current w.o.s
        # population_personality(TOTAL_POP, "population personality list") #print current personality rate

    ############# PRINT FINAL WORD ORDERS WITH A RANGE #############

    population_final_word_orders(TOTAL_POP, "population final word orders")  # print last 4 generations w.o.'s
    population_personality(TOTAL_POP, "population personality list")  # print last 4 generations personalities


main_simulation()
