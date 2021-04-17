pop_size = 30
num_generations = 50
ind_bit_len = 10
tournament_size = 5

mutations = num_generations * pop_size * ind_bit_len
cx = num_generations * pop_size
selection = num_generations * pop_size * tournament_size
shuffle = num_generations * pop_size

print(mutations + cx + selection + shuffle)