import pickle


def clear(filename):
    fobj = open(filename,"w")
    fobj.close()




def store(filename, rewards):

   
    fobj = open(filename, "wb")
    #obj = values(rewards)


    pickle.dump(rewards, fobj)
    fobj.close()

def read(filename):
    fobj = open(filename, "rb")
    result = []
   
    try:
        while True:
            obj = pickle.load(fobj)
            result.extend(obj)
            #obj is always of type array 

        
    except EOFError:
        return result
    return result

    fobj.close()