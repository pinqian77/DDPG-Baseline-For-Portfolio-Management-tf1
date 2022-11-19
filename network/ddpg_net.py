import tensorflow as tf
from network.build import CNN

class ActorNetwork(object):
    def __init__(self, sess, config, action_bound):
        """
        Args:
            sess: a tensorflow session
            config: a general config file
            action_bound: whether to normalize action in the end
        """
        self.sess = sess
        self.action_bound = action_bound
        self.config = config
        self.feature_number = config['input']['feature_number']
        self.action_dim = config['input']['asset_number'] + 1
        self.window_size = config['input']['window_size']
        self.learning_rate = config['training']['actor learning rate']
        self.tau = config['training']['tau']
        self.batch_size = config['training']['batch size']


        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                             tf.multiply(self.target_network_params[i], 1. - self.tau))
                                             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None] + [self.action_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                    self.learning_rate,
                                                    decay_steps=config['training']['episode']*config['training']['max step'],
                                                    decay_rate=0.96,
                                                    staircase=True)

        self.optimize = tf.keras.optimizers.Adam(self.lr_schedule).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        actor_net = CNN(self.feature_number,self.action_dim, self.window_size, self.config['actor_layers'])
        inputs = actor_net.input_tensor
        out = actor_net.output

        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)

        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        """
        Args:
            inputs: a observation with shape [None, action_dim, window_length, feature_number]
            a_gradient: action gradients flow from the critic network
        """
        inputs = inputs[:, :, -self.window_size:, :]
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })


    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, config, num_actor_vars):
        self.sess = sess
        self.config = config
        self.feature_number = config['input']['feature_number']
        self.action_dim = config['input']['asset_number'] + 1
        self.window_size = config['input']['window_size']
        self.learning_rate = config['training']['critic learning rate']
        self.tau = config['training']['tau']
        self.batch_size = config['training']['batch size']

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.target_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.keras.losses.mean_squared_error(self.target_q_value, self.out)


        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                    self.learning_rate,
                                                    decay_steps=config['training']['episode']*config['training']['max step'],
                                                    decay_rate=0.96,
                                                    staircase=True)

        self.loss_gradients = tf.gradients(self.loss, self.network_params)
        self.optimize = tf.keras.optimizers.Adam(
            self.lr_schedule).apply_gradients(zip(self.loss_gradients,self.network_params))

        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):

        critic_net = CNN(self.feature_number,self.action_dim,
                            self.window_size, self.config['critic_layers'])

        inputs = critic_net.input_tensor
        action = critic_net.predicted_w
        out = critic_net.output

        return inputs, action, out

    def train(self, inputs, action, target_q_value):
        """
        Args:
            inputs: observation
            action: predicted action
            target_q_value:
        """
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run([self.out, self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.target_q_value: target_q_value
        })

    def predict(self, inputs, action):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
