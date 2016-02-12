'''
test = np.array([[1, 2, 3, 4, 5, 6, 7, 8]]).view()

test.shape = [2] + list([2, 2])
#self.data[index] = data.view().reshape(self.shape)

print test
'''
for i, j in [(1, 4), (2, 5), (3, 6)]:
    print i
    print j


header = []
header += ['VARIABLES = "' + '" "'.join(('a', 'b', 'c')) + '"']

print header
