from itertools import product
from copy import deepcopy
import numpy as np

def getValues(data):
	# takes a parameter sweep specification in the form ["keyword", val1, val2, ...] and transforms it
	# to the appropriate set of parameter values
	#
	# inputs:
	#	data: a broom list specifying a parameter sweep; data[0] = keyword, data[1:] = specification
	# outputs:
	#	list of values that will be swept over
	
	keyword = data[0]	# per the specification, first element is a keyword
	specs = data[1:]	# rest of elements specify the parameters to sweep over

	if keyword == 'list':
		values = specs
	elif keyword == 'range':
		values = np.arange(*specs)
	elif keyword == 'linspace':
		values = np.linspace(*specs)

	return values
	
def addGroup(new_group, groups, locations, values):
	# Integrate a new group into the existing group/location/values structure
	# If the groups already exists, finds the index; if it is new, adds a new spot for it
	# Uses side-effects to do this
	#
	# inputs:
	#	new_group:	name of incoming group
	# outputs:
	#	idx:	index of where the new group was is location
	
	if new_group in groups and new_group != 0:
		idx = groups.index(new_group)
	else:
		groups.append(new_group)
		locations.append([])
		values.append([])
		idx = len(groups) - 1
	return idx


def getSweeps(d, l=[]):
	# takes a dictionary of parameter specifications and finds all parameters that are to have multiple
	# values for the parameter sweep
	#
	# inputs:
	# 	d:	dictionary with parameters in the broom format; nested dictionaries give a tree structure
	# 	l:	current location in the dictionary, used for recursively descending the dictionaries; not to be used
	#
	# outputs:
	#	groups: 	list of groups of variables that should be swept together; 0's are singleton groups
	#	locations:	list of lists (one per group) of the locations of the parameters in the tree
	#	values:		list of lists (one per gorup) of values for the parameters
	groups , locations, values = [], [], []
	for k in d:
		data = d[k]
		new_path = l + [k] # keep track of where you are in the tree
		
		# identify special sweep parameters -- a list with the first element as a string
		if type(data) is list and isinstance(data[0], basestring):
			
			# group given explicitly
			if data[0] == 'group':
				group_name= data[1]
				spec = data[2]
			
			# no group given -- add a dummy group indicated by 0
			else:
				group_name = 0
				spec = data
			
			idx = addGroup(group_name, groups, locations, values)
			locations[idx].append(new_path)
			values[idx].append(getValues(spec))

		# if element is another dictionary, descend the tree by recursing
		elif type(data) is dict:
			# recursive call
			new_groups, new_locations, new_values = getSweeps(data, new_path)
			# merge the values from subtree into current values
			for (grp, loc, val) in zip(new_groups, new_locations, new_values):
				if grp in groups and grp != 0:
					idx = groups.index(grp)
				else:
					groups.append(grp)
					locations.append([])
					values.append([])
					idx = len(groups) - 1
				locations[idx].extend(loc)
				values[idx].extend(val)

	return groups, locations, values


def setDict(dictionary, location, value):
	# a nested dictionary has as tree structure; this function sets the value at one of the leaves
	# uses side-effects to acheive this
	# input:
	#	dictionary: nested dictionary
	#	location: a list of keys that specifies a location in the tree
	#	value: the value to set at the leaf defined by the location
	# ouput:
	#	side-effect on dict

	reference = dictionary # we don't want to change the values in dictionary without changing what the name points do
	# descend the tree using the values in location
	for l in location[:-1]:
		reference = reference[l]
	# set the value at the leaf
	reference[location[-1]] = value

def makeDicts(baseDict, verbose=False):
	# given a base dictionary that specifies parameter sweeps, create new dictionaries -- one for each unique set
	# of paramters in the sweep
	# input:
	#	baseDict: the base dictionary in the corrent format
	#	verbose: optional, print info about inferred parameter settings
	# output:
	#	dicts: a list of dictionaries, one for each setting of the parameters
	#	inds: a list of tuples, also one for each setting; specifies which parameter values were used in that setting

	groups, locations, values = getSweeps(baseDict)

	if verbose:
		for (g, l, v) in zip(groups, locations, values):
			print 'group:\t' + str(g)
			print 'location:\t' + str(l)
			print 'value:\t' + str(zip(*v))
		print '----------'

	# for groups with more than one parameter, values will be swept together -- group them as such
	values = map(lambda v: zip(*v), values)
	# associate each possible parameter set within a group with an index
	values = map(lambda v: zip(range(len(v)), v), values)

	dicts = []
	inds = []

	# compute all unique combinations of parmeter values with cartesian product
	for paramSet in product(*values):						# for each setting of the parameters
		# copy the dictionary
		d = deepcopy(baseDict)
		# separate indices from values
		idx, paramValues = zip(*paramSet)
		if verbose:
			print idx
		for (locs, vals) in zip(locations, paramValues):	# for each group of parameters
			for (loc, val) in zip(locs, vals):				# for each individual parameter in each group
				# set the appropriate values in the dictionary
				setDict(d, loc, val)
				if verbose:
					print str(loc) + '\t' + str(val)
		dicts.append(d)
		inds.append(idx)

	return inds, dicts, locations, values

class Bunch(object):
    '''
    A class that wraps a nested dictionary and provides attribute-style access
    '''
    
    def __init__(self, adict):
        self.__dict__.update(adict)
        for (k, v) in self.__dict__.iteritems():
            if type(v) is dict:
                self.__dict__[k] = Bunch(v)
    
    def __repr__(self):
        return self.__dict__.__str__()

def loadParams(filename):
	f = open(filename, 'r')
	return Bunch(json.loads(f.read()))