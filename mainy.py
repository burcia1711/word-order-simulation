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

k = 1  # factor
n = 25  # number of people in one starting group

error = 10  # add to the weights
personality_weight = [5, 5]
coeff_stubborn = 1000
coeff_flexible = 100

error_or_pressure_rate = 0.00001  # mesh
# 0.000001  # star
# 0.0000001  # one-to-one

GEN = 5
BIAS_TYPE = "uniform"
uniformIRREV = []
uniformREV = []
REV_BIAS = [1, 1, 1, 1, 1, 5]
IRREV_BIAS = [1, 5, 1, 1, 1, 1]

i_irrev = random.randint(0, 5)
i_rev = random.randint(0, 5)

REV_BIAS = [1 if i is not i_rev else 5 for i in range(6)]
IRREV_BIAS = [1 if i is not i_irrev else 5 for i in range(6)]


tendency = [3, 8, 410, 350, 20, 70]  # world's current distribution
TEND_COM = random.choices(basic_orders, weights=tendency, k=1)
TEND_REV = 'SVO'  # what we know
TEND_IRREV = 'SOV'  # what we know
TEND = True


def form_irrev_weight(personality, bias_type):
    if bias_type == "uniform":
        if personality == "F":
            return [coeff_flexible] * 6
        elif personality == "S":
            return [coeff_stubborn] * 6
    elif bias_type == "biased":
        if personality == "F":
            return [i * coeff_flexible for i in IRREV_BIAS]
        elif personality == "S":
            return [i * coeff_stubborn for i in IRREV_BIAS]
    elif bias_type == "random":
        if personality == "F":
            return [i * coeff_flexible for i in random.sample(range(0, 10), 6)]
        elif personality == "S":
            return [i * coeff_stubborn for i in random.sample(range(0, 10), 6)]


def form_rev_weight(personality, bias_type):
    if bias_type == "uniform":
        if personality == "F":
            return [coeff_flexible] * 6
        elif personality == "S":
            return [coeff_stubborn] * 6
    elif bias_type == "biased":
        if personality == "F":
            return [i * coeff_flexible for i in REV_BIAS]
        elif personality == "S":
            return [i * coeff_stubborn for i in REV_BIAS]
    elif bias_type == "random":
        if personality == "F":
            return [i * coeff_flexible for i in random.sample(range(0, 10), 6)]
        elif personality == "S":
            return [i * coeff_stubborn for i in random.sample(range(0, 10), 6)]


class Agent:
    def __init__(self, g, p, mother=None, father=None):  # First agent has no parent, children will.
        self.generation = g
        self.personality = p
        if mother is None and father is None:  # first generation, starting point
            self.irrev_weights = form_irrev_weight(p, BIAS_TYPE)
            self.rev_weights = form_rev_weight(p, BIAS_TYPE)
        else:  # if it is a child, mother & father should affect
            self.irrev_weights = self.list_average(form_irrev_weight(p, BIAS_TYPE), self.set_irrev_weights(mother, father))
            self.rev_weights = self.list_average(form_rev_weight(p, BIAS_TYPE), self.set_rev_weights(mother, father))

    def new_weight_with_pressure_irrev(self, order):  # some pressures made us eliminate others
        weight = []
        for i in basic_orders:
            if TEND:
                if i == order or i == TEND_COM[0]:
                    weight.append(error_or_pressure_rate * (error ** k) * GEN)  # add 1 to the used word order
                else:
                    weight.append(-error_or_pressure_rate * (error ** k) * GEN)  # add -1 to weights of non-used word orders
            else:
                if i == order:
                    weight.append(error_or_pressure_rate * (error ** k) * GEN)  # add 1 to the used word order
                else:
                    weight.append(-error_or_pressure_rate * (error ** k) * GEN)  # add -1 to weights of non-used word orders

        return weight

    def new_weight_with_pressure_rev(self, order):  # some pressures made us eliminate others
        weight = []
        for i in basic_orders:
            if TEND:
                if i == order or i == TEND_COM[0]:
                    weight.append(error_or_pressure_rate * (error ** k) * GEN)  # add 1 to the used word order
                else:
                    weight.append(-1 * error_or_pressure_rate * (error ** k) * GEN)  # add -1 to weights of non-used word orders
            else:
                if i == order:
                    weight.append(error_or_pressure_rate * (error ** k) * GEN)  # add 1 to the used word order
                else:
                    weight.append(-1 * error_or_pressure_rate * (error ** k) * GEN)  # add -1 to weights of non-used word orders
        return weight

    def list_summation(self, l1, l2):  # adding two lists
        res_lt = []
        for x in range(len(l1)):
            if l1[x] <= 0:
                res_lt.append(0)
            else:
                res_lt.append(l1[x] + l2[x])
        return res_lt

    def list_average(self, l1, l2):  # for averaging mother and father's weights
        res_lt = [(l1[x] + l2[x]) for x in range(len(l1))]
        return res_lt

    def set_irrev_weights(self, mother, father):  # calculate average of mother+father weights for irreversible sentences
        return self.list_average(mother.irrev_weights, father.irrev_weights)

    def set_rev_weights(self, mother, father):  # calculate average of mother+father weights for reversible sentences
        return self.list_average(mother.rev_weights, father.rev_weights)

    def add_rev_weights_with_pressure(self, word_order):
        self.rev_weights = self.list_summation(self.rev_weights, self.new_weight_with_pressure_rev(word_order))

    def add_irrev_weights_with_pressure(self, word_order):
        self.irrev_weights = self.list_summation(self.irrev_weights, self.new_weight_with_pressure_irrev(word_order))


def make_first_gen_agents(N):  # create N number of agents with different random personalities
    gen = 1
    population = []

    for i in range(N):
        p = random.choices([0, 1], weights=personality_weight, k=1)[0]
        agent = Agent(gen, personalities[p])
        population.append(agent)

    print(TEND_COM)
    return population


def create_children(mother, father,
                    number_of_children):  # create children of given mother and father, with the given number of children
    children = []
    for i in range(number_of_children):
        p = random.choices([0, 1], weights=personality_weight, k=1)[0]
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
    return average_children


def create_generation(prev_generation_pop):
    population_length = len(prev_generation_pop)
    pairs = create_pairs(prev_generation_pop)
    next_gen = []
    for p in pairs:
        children_number = calculate_average_children_number_per_family(population_length)
        next_gen.extend(create_children(prev_generation_pop[p[0]], prev_generation_pop[p[1]], children_number))
    return next_gen


def plot_freq_list(lst, ttle, id):
    count = Counter(sorted(lst))
    df = pandas.DataFrame.from_dict(count, orient='index')
    df.plot(kind='bar', color="orange")
    plt.title("%s %d %s" % (ttle, id, TEND_COM[0] if TEND else ""))
    if len(ttle.split()) > 3:
        if ttle.split()[1] == "first":
            if ttle.split()[4] == "irrev":
                f = "1"
            else:
                f = "2"
        else:
            if ttle.split()[4] == "irrev":
                f = "3"
            else:
                f = "4"

        plt.savefig('auto-tests/' + str(id) + "-" + f + '.png')
    else:
        plt.savefig('auto-tests/' + str(id) + "-" + "5" + '.png')
    #plt.show()


def generate_word_order_list(order_list, weight, n):
    return random.choices(order_list, weights=weight, k=n)


def make_utterance(n):
    return random.choices(sentence_type, weights=sentence_weights, k=n)


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
def n_people_communicate(n_p, n_sent,
                         population):  # n_people is the number of people, n_sent is the # of sentences for the communication, population is the given population
    n_people = min(n_p, len(population))
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
                    population[l].add_irrev_weights_with_pressure(spoken_word_order[0])

        else:
            spoken_word_order = generate_word_order_list(basic_orders, population[speaker].rev_weights,
                                                         1)  # generate a word order for given sentence
            # print(f'rev: {spoken_word_order}')
            # update listeners
            for l in selected_people:
                if l != speaker:
                    population[l].add_rev_weights_with_pressure(spoken_word_order[0])


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


def population_final_word_orders(population, ttle, id):
    IRREV_LIST = []
    REV_LIST = []

    for people in population:
        IRREV_LIST.extend(generate_word_order_list(basic_orders, people.irrev_weights, 100))
        REV_LIST.extend(generate_word_order_list(basic_orders, people.rev_weights, 100))

    plot_freq_list(IRREV_LIST, ttle + " irrev", id)
    plot_freq_list(REV_LIST, ttle + " rev", id)


def population_personality(population, ttle, id):
    personality_list = []

    for people in population:
        personality_list.extend(people.personality)

    plot_freq_list(personality_list, ttle, id)


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


def main_simulation(bias, gen, tenden, comSize, network, dataSize, personality, id):
    global BIAS_TYPE
    global TEND
    global personality_weight
    global error_or_pressure_rate
    global GEN
    global REV_BIAS
    global IRREV_BIAS
    global TEND_COM
    global i_rev
    global i_irrev
    global tendency

    i_irrev = random.randint(0, 5)
    i_rev = random.randint(0, 5)

    TEND_COM = random.choices(basic_orders, weights=tendency, k=1)

    TEND = tenden
    personality_weight = personality
    GEN = gen

    REV_BIAS = [1 if i is not i_rev else 2 for i in range(6)]
    print(REV_BIAS)
    IRREV_BIAS = [1 if i is not i_irrev else 2 for i in range(6)]
    print(IRREV_BIAS)

    if bias == "uniform":
        BIAS_TYPE = "uniform"
    elif bias == "biased":
        BIAS_TYPE = "biased"
    elif bias == "random":
        BIAS_TYPE = "random"

    TOTAL_POP = []
    population_first_gen = []
    population_first_gen.extend(make_first_gen_agents(comSize))  # create first community with 20 agents
    population_final_word_orders(population_first_gen, "population first word orders", id)
    group_communication_all_population_speaks(1000, population_first_gen)
    TOTAL_POP.extend(population_first_gen)
    last_gen = population_first_gen

    for i in range(gen):  # create gen generations
        new_gen = create_generation(last_gen)  # create a new generation from last generation
        last_gen = new_gen  # make new generation last generation
        TOTAL_POP.extend(new_gen)  # extend total population with last generation
        TOTAL_POP = list(
            filter(lambda person: person.generation >= i - 4, TOTAL_POP))  # filter out except last 4 generations
        if network == "mesh":
            error_or_pressure_rate = 0.00001
            group_communication_all_population_speaks(dataSize, TOTAL_POP)
        elif network == "star":
            error_or_pressure_rate = 0.000001
            n_groups_communicate(100, comSize, dataSize, TOTAL_POP)
        elif network == "one-to-one":
            error_or_pressure_rate = 0.0000001
            n_people_communicate(comSize, dataSize, TOTAL_POP)


    ############# PRINT FINAL WORD ORDERS WITH A RANGE #############

    population_final_word_orders(TOTAL_POP, "population final word orders", id)  # print last 4 generations w.o.'s
    population_personality(TOTAL_POP, "population personality list", id)  # print last 4 generations personalities
    print(BIAS_TYPE, TEND, personality_weight)

BIAS = ["uniform", "biased", "random"]
generations = [5, 10, 20]
tendencyWO = [True, False]
firstComSize = [1, 2, 4]
networks = ["mesh", "star", "one-to-one"]
data = [1000, 5000]
personalityCase= [[8, 2], [5, 5], [2, 8]]
id = 1
LEFT = 0
personalityCase = [[8, 2], [5, 5], [2, 8]]
id = 1
LEFT = 561

for bias in BIAS:
    for generation in generations:
        for tend in tendencyWO:
            for k in firstComSize:
                for network in networks:
                    for dt in data:
                        for personalWeight in personalityCase:
                            if id > LEFT:
                                print(bias, generation, tend, k*n, network, dt, personalWeight, id)
                                main_simulation(bias, generation, tend, k*n, network, dt, personalWeight, id)
                            id += 1

