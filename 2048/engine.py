"""
2048 Game Engine 
"""
import itertools
import numpy as np
import random

class GameEngine():
	""""""
	MATRIX_SIZE = 4
	ACTIONS = ["left", "right", "up", "down", "undo"]
	POSSIBLE_INITIAL_VALUES = [2]
	POSSIBLE_NEW_VALUES = [2,4]

	def __init__(self):
		""""""
		self.matrix = np.zeros(
			(self.MATRIX_SIZE, self.MATRIX_SIZE), 
			dtype=int
		)
		indices = self.get_random_empty_index(2)
		for idx, idy in indices:
			self.matrix[idx, idy] = random.choice(self.POSSIBLE_INITIAL_VALUES)
		self.score = 0
		self.num_moves = 0
		self.game_states = []
		# self.save_game_state()
		print(self.game_states)

	def get_random_empty_index(self, num=1):
		""""""
		idxs, idys = np.where(self.matrix == 0)
		indices = list(zip(idxs, idys))
		if num > 1:
			return random.sample(indices, num)
		else:
			return random.choice(indices)

	def add_new_value(self):
		""""""
		index = self.get_random_empty_index()
		self.matrix[index] = random.choice(self.POSSIBLE_NEW_VALUES)

	def action(self, action_label):
		""""""
		assert action_label in self.ACTIONS

		print(action_label)
		getattr(self, action_label)()
		if self.game_states:
			self.add_new_value()	
			self.is_complete()
		# print(self.game_states)
		self.num_moves += 1
		return self.matrix, self.score, self.num_moves

	def stack(self):
		""""""

		def move_nonzero_to_left(row):
			""""""
			nonzero_indices = np.nonzero(row)[0]
			row[:len(nonzero_indices)] = row[nonzero_indices]
			row[len(nonzero_indices):] = 0
			return row

		# Apply the function to each row of the matrix
		np.apply_along_axis(
			move_nonzero_to_left, 
			axis=1, 
			arr=self.matrix
		)

	def combine(self):
		""""""
		def add_consecutive_identical_values(row):
			""""""
			for jdx in range(len(row)-1):
				if row[jdx] and row[jdx] == row[jdx+1]:
					row[jdx] *= 2
					row[jdx+1] = 0
					self.score += row[jdx]
			return row 

		# Apply the function to each row of the matrix
		np.apply_along_axis(
			add_consecutive_identical_values, 
			axis=1, 
			arr=self.matrix
		)

	def reverse(self):
		self.matrix = np.flip(self.matrix, axis=1)

	def transpose(self):
		self.matrix = self.matrix.T

	def save_game_state(self):
		self.game_states.append((self.matrix.copy(), self.score))

	def left(self):
		self.save_game_state()
		self.stack()
		self.combine()
		self.stack()

	def right(self):
		self.save_game_state()
		self.reverse()
		self.stack()
		self.combine()
		self.stack()
		self.reverse()
		
	def up(self):
		self.save_game_state()
		self.transpose()
		self.stack()
		self.combine()
		self.stack()
		self.transpose()	

	def down(self):
		self.save_game_state()
		self.transpose()
		self.reverse()
		self.stack()
		self.combine()
		self.stack()
		self.reverse()
		self.transpose()

	def undo(self):
		self.matrix, self.score = self.game_states.pop()


	def check_adjacent_values(self):
		for idx in range(self.MATRIX_SIZE):
			for jdx in range(self.MATRIX_SIZE-1):
				if self.matrix[idx, jdx] == self.matrix[idx, jdx+1]:
					return True

		for idx in range(self.MATRIX_SIZE-1):
			for jdx in range(self.MATRIX_SIZE):
				if self.matrix[idx, jdx] == self.matrix[idx+1, jdx]:
					return True
		return False

	def is_complete(self):
		if 2048 in self.matrix:
			print("WIN")
			return True

		if 0 not in self.matrix and not self.check_adjacent_values():
			print("LOSS")
			return True

		return False



