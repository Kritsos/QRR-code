Code implementation of the Quantized Rank Reduction scheme. All required python libraries are in requirements.txt
and can be installed using

<pre>
pip install -r requirements.txt
</pre>

The scripts use pytorch and flwr to run the federated learning simulation. The behavior of the server and clients
for each method are defined as separate python class, e.g. for SGD the server behavior is defined in sgd.py and the
client behavior is defined in sgd_client.py.

In sim_config.py you can change all the parameters and hyperparameters of the methods, as well as specify the dataset
to use in the simulation. In utils.py you can define the model architecture you wish to use.

Once the simulation configuration is done you can run

<pre>
python run_simulation.py
</pre>

to run the simulation. Once the simulation is over the results and figures will automatically be saved in the current directory.
