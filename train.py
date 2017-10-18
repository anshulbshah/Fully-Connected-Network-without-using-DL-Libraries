import cPickle, gzip
from numpy import *
import argparse
import elastic

seed=1234
random.seed(seed)

parser = argparse.ArgumentParser() 
parser.add_argument("--lr",type=float)
parser.add_argument("--momentum",type=float,default=0.0)
parser.add_argument("--num_hidden" , type=int)
parser.add_argument("--sizes")
parser.add_argument("--activation")
parser.add_argument("--loss")
parser.add_argument("--opt")
parser.add_argument("--batch_size",type=int)
parser.add_argument("--anneal")
parser.add_argument("--save_dir")
parser.add_argument("--expt_dir")
parser.add_argument("--mnist")
args = parser.parse_args()

#elastic.mainfun(args.mnist)

f = gzip.open(args.mnist, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

'''
f1 = gzip.open('mnist.pkl.gz','rb')
train_set = cPickle.load(f)
f1.close
'''
sizes = map(int,args.sizes.split(','))

def sigmoid(x):
	return 1.0/(1.0+exp(-x))

def sigmoid_prime(x):
	return sigmoid(x)*(1.0-sigmoid(x))

def tanh1(x):
	return tanh(x)

def tanh_prime(x):
	return 1.0 - (tanh(x))**2

def softmax(w):
    e = exp(w - amax(w,0))
    dist = e / sum(e,0)
    return dist

eta = args.lr

log_train = open(args.expt_dir+'log_loss_train.txt','w')
log_valid = open(args.expt_dir+'log_loss_valid.txt','w')
log_test = open(args.expt_dir+'log_loss_test.txt','w')
err_train = open(args.expt_dir+'log_err_train.txt','w')
err_valid = open(args.expt_dir+'log_err_valid.txt','w')
err_test = open(args.expt_dir+'log_err_test.txt','w')
pred_test = open(args.expt_dir+'test_predictions.txt','w')
pred_valid = open(args.expt_dir+'valid_predictions.txt','w')
#plotfile_test = open(args.expt_dir+'error_test.txt','w')
#plotfile_valid = open(args.expt_dir+'error_valid.txt','w')
#plotfile_train = open(args.expt_dir+'error_train.txt','w')


class NeuralNet(object):
		def __init__(self, args):
			self.sizes=[784]
			self.sizes.extend(sizes)
			self.sizes.append(10)
			self.biases = [2*random.random([y, 1])-1 for y in self.sizes[1:]]
			self.weights = [random.uniform(-1*sqrt(6.0/(x+y)),1*sqrt(6.0/(x+y)),[y, x]) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
			self.num_layer = args.num_hidden + 1
			self.errorv_old =10**8
			self.flag=0

		def Opti(self, training_data, epochs, mini_batch_size, eta,mom,
		    test_data=test_set):
		    if test_data: n_test = len(test_data)
		    n = len(training_data[0])
		    if(mini_batch_size%5!=0 and mini_batch_size !=1):
		    	print 'Why you no check batch size!'
		    	return -1
		    for j in xrange(epochs):
			if(j==25):
				fpickle = open(args.save_dir+'weightsandbiases.pkl','w')
				cPickle.dump([self.weights,self.biases],fpickle)
				fpickle.close()
			if(self.flag and args.anneal=='true'):
				if(eta>0.0000001):
					eta = eta/1.2
				self.weights = weights_old
				self.biases = biases_old
			c=range(0,n,1)
			random.shuffle(c)
			mini_batches = [training_data[0][c[k:k+mini_batch_size]] for k in xrange(0, n, mini_batch_size)]
			mini_batches_y = [training_data[1][c[k:k+mini_batch_size]] for k in xrange(0, n, mini_batch_size)]
			if(args.opt!='adam'):
				grad_b_prev = [zeros(b.shape) for b in self.biases]
				grad_w_prev = [zeros(w.shape) for w in self.weights]
			if(args.opt=='adam'):
				mb_prev = [zeros(b.shape) for b in self.biases]
				mw_prev = [zeros(w.shape) for w in self.weights]
				vb_prev = [zeros(b.shape) for b in self.biases]
				vw_prev = [zeros(w.shape) for w in self.weights]
			print('epoch ',j)
			beta1 = 0.9
			beta2 = 0.999
			# lam=0.1
		        t = 0
		        for mini_batch in mini_batches:
					activation = (mini_batch.T)
					activations = [(mini_batch).T]
					zs = []
					for b, w in zip(self.biases, self.weights):
						z = dot(w, activation)+b
						zs.append(z)
						activation = self.actifunc(z)
						activations.append(activation)
					y_hat = softmax(z)
					if(args.opt=='nag'):
						self.weights = [w+(mom/mini_batch_size)*nwp for w,nwp in zip(self.weights,grad_w_prev)] 
						self.biases = [b+(mom/mini_batch_size)*nbp for b,nbp in zip(self.biases, grad_b_prev)]

					grad_b,grad_w = self.backprop(zs,activations,y_hat,j,mini_batches_y[t].T,mini_batch_size)	
					if(args.opt=='adam'):
						mw = [multiply(beta1,kmw_prev) + multiply((1-beta1),kgrad_w/1.0) for kmw_prev,kgrad_w in zip(mw_prev,grad_w)]
						vw = [multiply(beta2,kvw_prev) + multiply((1-beta2),(kgrad_w/1.0)**2) for kvw_prev,kgrad_w in zip(vw_prev,grad_w)]
						self.weights = [w-eta*(nmw_prev/(1-beta1**(t+1)))/(sqrt((nvw_prev/(1-beta2**(t+1)))+(10**(-8)))) for w,nmw_prev,nvw_prev in zip(self.weights,mw_prev,vw_prev)]
						mb = [multiply(beta1,kmb_prev) + multiply((1-beta1),kgrad_b/1.0) for kmb_prev,kgrad_b in zip(mb_prev,grad_b)]
						vb = [multiply(beta2,kvb_prev) + multiply((1-beta2),(kgrad_b/1.0)**2) for kvb_prev,kgrad_b in zip(vb_prev,grad_b)]
						self.biases = [b-eta*(nmb_prev/(1-beta1**(t+1)))/(sqrt((nvb_prev/(1-beta2**(t+1)))+(10**(-8)))) for b,nmb_prev,nvb_prev in zip(self.biases, mb_prev,vb_prev)] 
					if(args.opt!='adam'):
						self.weights = [w-(eta/mini_batch_size)*nw-(mom/mini_batch_size)*nwp for w, nw,nwp in zip(self.weights, grad_w,grad_w_prev)] 
						# self.weights = [w*(1-(eta*lam)/mini_batch_size)-(eta/mini_batch_size)*nw-(mom/mini_batch_size)*nwp for w, nw,nwp in zip(self.weights, grad_w,grad_w_prev)] 
						self.biases = [b-(eta/mini_batch_size)*nb-(mom/mini_batch_size)*nbp for b, nb,nbp in zip(self.biases, grad_b,grad_b_prev)] 
						if(args.opt!='gd'):
							grad_b_prev = grad_b
							grad_w_prev = grad_w
					if(args.opt=='adam'):
						mw_prev,vw_prev,mb_prev,vb_prev = mw,vw,mb,vb
                                        if(t%100==0 or t==(len(train_set[0])/mini_batch_size)-1):
                                                self.flag=self.evaluate(test_data,j,t,eta,mini_batch_size)				
					t = t + 1
		    	if(self.flag!=1):
                    		weights_old = self.weights
                        	biases_old = self.biases		

		def backprop(self,zs,activations,y_hat,j,y,batch_size):
			delta = self.cost_derivative(y_hat, y)
			grad_b_back = [zeros(b.shape) for b in self.biases]
			grad_w_back = [zeros(w.shape) for w in self.weights]
			for k in xrange(self.num_layer,0,-1):
				grad_w_back[k-1] = (dot(delta,activations[k-1].T))
				grad_b_back[k-1] = reshape(sum(delta,1),(delta.shape[0],1))
				if(k !=1):
					grad_h = (dot(self.weights[k-1].T,delta))
					delta = (grad_h*self.actifunc_prime((zs[k-2])))
			return (grad_b_back, grad_w_back)

		def evaluate(self, test_data,epoch_number,mini_batch_number,eta_passed,mini_size):
			perror=0
			test_batch = test_data[0]
			test_batch_y = test_data[1]
			valid_batch = valid_set[0]
			valid_batch_y = valid_set[1]
			if(mini_batch_number==0):
				self.d = range(0,50000,1)
				random.shuffle(self.d)
			train_batch=train_set[0][self.d[0:10000]]
			train_batch_y = train_set[1][self.d[0:10000]]
			to_test = concatenate((test_batch ,valid_batch,train_batch),axis = 0)
			to_test_labels = concatenate((test_batch_y ,valid_batch_y,train_batch_y),axis = 0)
			activationeval = (to_test.T)
			vars1 = 0
			for b, w in zip(self.biases, self.weights):
				if(vars1 == args.num_hidden):
					zseval = dot(w, activationeval)+b
				activationeval = self.actifunc(dot(w,activationeval) + b)
				vars1 = vars1 + 1
			y_hat = softmax(zseval)
			t1 = argmax(y_hat[:,:10000],0)
			t2 = argmax(y_hat[:,10000:20000],0)
			perror = where(argmax(y_hat[:,:10000],0)==to_test_labels[:10000])[0].shape[0]
			perror_valid = where(argmax(y_hat[:,10000:20000],0)==to_test_labels[10000:20000])[0].shape[0]
			perror_train = where(argmax(y_hat[:,20000:],0)==to_test_labels[20000:])[0].shape[0]
			error = 0
			errorv = 0
			errort = 0
			oo = 0
			if(epoch_number==49):
				for item in t1:
					pred_test.write(str(item) +'\n')
				for items in t2:
					pred_valid.write(str(items) + '\n')
			if(args.loss=='ce'):
				for k in to_test_labels[:10000]:
					error = error - log(y_hat[k,oo])
					oo = oo + 1
				for m in to_test_labels[10000:20000]:
					errorv = errorv - log(y_hat[m,oo])
					oo = oo + 1
				for r in to_test_labels[20000:]:
					errort = errort - log(y_hat[r,oo])
					oo = oo + 1		
			else:
				new_y_mse = zeros((y_hat).shape)
				for h in range((y_hat).shape[1]):
					new_y_mse[to_test_labels[h],h] = 1			
				error = 0.5*sum((y_hat[:,:10000]-new_y_mse[:,:10000])**2)
				errorv = 0.5*sum((y_hat[:,10000:20000]-new_y_mse[:,10000:20000])**2)
				errort = 0.5*sum((y_hat[:,20000:]-new_y_mse[:,20000:])**2)
							
			# print(perror/10000.00,error/10000.00)
			# print(perror_valid/10000.00,errorv/10000.00)
			# print(perror_train/10000.00,errort/10000.00)
			# plotfile_test.write("{} \n".format(error))
			# plotfile_valid.write("{} \n".format(errorv))
			# plotfile_train.write("{} \n".format(errort))

			if(mini_batch_number!=(len(train_set[0])/mini_size)-1):
				log_test.write("Epoch {}, Step {}, Loss: {:.5f}, lr: {} \n".format(epoch_number,mini_batch_number,error/10000.0,eta_passed))
				log_valid.write("Epoch {}, Step {}, Loss: {:.5f}, lr: {} \n".format(epoch_number,mini_batch_number,errorv/10000.0,eta_passed))
				log_train.write("Epoch {}, Step {}, Loss: {:.5f}, lr: {} \n".format(epoch_number,mini_batch_number,errort/10000.0,eta_passed))
				err_test.write("Epoch {}, Step {}, Error: {:.2f}, lr: {} \n".format(epoch_number,mini_batch_number,(10000.0-perror)/100.0,eta_passed))
				err_valid.write("Epoch {}, Step {}, Error: {:.2f}, lr: {} \n".format(epoch_number,mini_batch_number,(10000.0-perror_valid)/100.0,eta_passed))
				err_train.write("Epoch {}, Step {}, Error: {:.2f}, lr: {} \n".format(epoch_number,mini_batch_number,(10000.0-perror_train)/100.0,eta_passed))
			if(mini_batch_number==(len(train_set[0])/mini_size)-1):
				if(errorv<self.errorv_old):
					self.errorv_old = errorv
					return 0
				else:
					return 1
			else:
				return self.flag			




		def cost_derivative(self, output_activations, y):
			new_y = zeros((output_activations).shape)
			new_y2 = zeros((output_activations).shape)
			for h in range((output_activations).shape[1]):
				new_y[y[h],h] = 1
			if(args.loss=='ce'):
				return (output_activations-new_y)
			else:
				return output_activations*(output_activations-new_y)*(ones(output_activations.shape)-output_activations)
			

		def actifunc(self,x):
				if(args.activation=='sigmoid'):
					return sigmoid(x)
				else:
					return tanh(x)

		def actifunc_prime(self,x):
			if args.activation=='sigmoid':
				return sigmoid_prime(x)
			else:
				return tanh_prime(x)				


NN = NeuralNet(args)
NN.Opti(train_set, 50, args.batch_size, eta,args.momentum)
