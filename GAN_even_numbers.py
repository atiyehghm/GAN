'''
training a super simple GAN for generating even numbers

the main article is:
https://towardsdatascience.com/build-a-super-simple-gan-in-pytorch-54ba349920e4
'''
import numpy as np
import math
import torch
import torch.nn as nn
from typing import List
import mlflow
import mlflow.pytorch

def convert_float_matrix_to_int_list(
    float_matrix: np.array, threshold: float = 0.5
) -> List[int]:
    return [
        int("".join([str(int(y)) for y in x]), 2) for x in float_matrix >= threshold
    ]



def binary_number(number):
    if number < 0 or type(number) is not int:
        raise ValueError("Only Positive integers are allowed")

    return [int(x) for x in list(bin(number))[2:]]

print(binary_number(12))

def generate_even_numbers(max_number, batch_size):
    max_len = int(math.log(max_number, 2))
    samples = np.random.randint(0, int(max_number/2), batch_size)
    labels = [1 for i in range(batch_size)]
    data = [binary_number(int(x*2)) for x in samples]
    ##equalization of the size
    data = [[0]*(max_len-len(x))+x for x in data]
    return data, labels


data,  labels = generate_even_numbers(128, 13)
print(data, labels)
#

class Generator(nn.Module):

    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))


class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_length), 1);
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))



experiment_id = mlflow.create_experiment('GAN4')
# print(type(experiment_id))
def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500):

    input_length = int(math.log(max_int, 2))

    # Models
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise)

        # Generate examples of even real data
        true_data, true_labels = generate_even_numbers(max_int, batch_size=batch_size)
        true_labels = torch.tensor(true_labels).resize(batch_size,1).float()
        true_data = torch.tensor(true_data).float()

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        generator_discriminator_out = discriminator(generated_data)
        generator_loss = loss(generator_discriminator_out, true_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size,1))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()
        mlflow.log_param('generator lr',0.001)
        mlflow.log_param('discriminator lr', 0.001)
        mlflow.log_metric('generator loss', generator_loss.item())
        mlflow.log_metric('discriminator loss', discriminator_loss.item())

        if i % 100 == 0:
            print(str(i) +" :", convert_float_matrix_to_int_list(generated_data,0.5))
    print(str(i) + " :", convert_float_matrix_to_int_list(generated_data, 0.5))
    mlflow.pytorch.log_model(generator, 'generator')
    mlflow.pytorch.log_model(discriminator, 'discriminator')


mlflow.set_experiment(int(experiment_id))
train(128,10,500)
