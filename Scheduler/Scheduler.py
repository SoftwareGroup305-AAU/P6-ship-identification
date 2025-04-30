import math


#Nede i bunden er der en HowTo om hvordan man implementere det.



#Custom_lr
def custom_lr(epoch, start_lr=0.001, max_lr=0.01, warmup_epochs=5, plateau_epochs=80, final_lr=0.0001):
    if epoch < warmup_epochs:
        return start_lr + (max_lr - start_lr) * (epoch / warmup_epochs)
    elif epoch < plateau_epochs:
        return max_lr
    elif epoch < (plateau_epochs + 30): #Hvis der er 2 plateau_epochs
        return start_lr
    else:
        return final_lr

def step_lr(epoch, start_lr=0.01, step1=10, step2=50, final_lr=0.0001):
    if epoch < step1:
        return start_lr
    elif epoch < step2:
        return 0.001
    else:
        return final_lr

def cosine_lr(epoch, total_epochs=100, max_lr=0.01, min_lr=0.0001):
    # Cosine annealing LR
    return 0.5 * (1 + math.cos(math.pi * epoch / total_epochs)) * (max_lr - min_lr) + min_lr



def cosine_delay_lr(epoch, total_epochs=100, max_lr=0.01, min_lr=0.0001, delay_epochs=10):
    """
    Cosine annealing with a delay in the beginning
    """
    if epoch < delay_epochs:
        return min_lr  # Keep the learning rate constant during delay period
    else:
        # Cosine annealing after delay period
        return 0.5 * (1 + math.cos(math.pi * (epoch - delay_epochs) / (total_epochs - delay_epochs))) * (max_lr - min_lr) + min_lr

# SGD_lr (Stochastic Gradient Descent)
def sgd_lr(optimizer, epoch, base_lr=0.01, decay_factor=0.1, step_size=30):
    """
    Basic Stochastic Gradient Descent with learning rate decay
    """
    lr = base_lr * (decay_factor ** (epoch // step_size))  # Learning rate decay every `step_size` epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def sdr_lr(epoch, initial_lr=0.01, sdr_factor=2, restart_epoch=30, final_lr=0.0001):
    """
    Step Decay with Restart (SDR) 
    """
    cycle = math.floor(1 + epoch / (2 * restart_epoch))
    x = abs(epoch / restart_epoch - 2 * cycle + 1)
    lr = initial_lr / math.pow(sdr_factor, cycle - 1)  # Decay after each cycle
    if epoch > restart_epoch:
        return lr * x  # Cosine-like restart within the cycle
    else:
        return initial_lr



# Dictionary to easily switch between schedulers
schedulers = {
    "custom_lr": custom_lr,
    "step_lr": step_lr,
    "cosine_lr": cosine_lr,
    "cosine_delay_lr": cosine_delay_lr,
    "sgd_lr": sgd_lr,
    "sdr_lr": sdr_lr
}

def get_lr(scheduler_name, epoch, total_epochs=100, **kwargs):
    return schedulers[scheduler_name](epoch, total_epochs=total_epochs, **kwargs)


# Smid det her ind i dit træningsloop:
'''
for epoch in range(epochs):
    # Hent den aktuelle læringsrate baseret på den valgte scheduler (fx "custom_lr")
    lr = get_lr(scheduler_name="custom_lr", epoch=epoch, total_epochs=epochs, start_lr=0.001, max_lr=0.01, warmup_epochs=5, plateau_epochs=80, final_lr=0.0001)
    
    # Opdater læringsraten i optimizerens parametre
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

'''