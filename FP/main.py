import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from sklearn.feature_selection import SelectPercentile as sp, f_classif


cutoff_percentile = 5 #for classifers
down_sample_size = (19,19) #image downampling size


#bounding-box class
class bounding_box:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def generate_features(self, integral):        
        return integral[self.y+self.height][self.x+self.width] + integral[self.y][self.x] - (integral[self.y+self.height][self.x]+integral[self.y][self.x+self.width])

#weak classifier Class
class weakClassifer:
    def __init__(self, pos_box, neg_box, thresh, pol):
        self.pos_box = pos_box
        self.neg_box = neg_box
        self.threshold = thresh
        self.pol = pol
    
    def classify_weak(self, x):
        feature = lambda ii: sum([pos.generate_features(ii) for pos in self.pos_box]) - sum([neg.generate_features(ii) for neg in self.neg_box])
        return 1 if self.pol* feature(x) < self.pol * self.threshold else 0



#Model class
class Model:
    def __init__(self, n_classifiers = 10):
        self.n_classifiers = n_classifiers
        self.clfs = []
        self.learning_rates = []
        

    def train(self, faces, non_faces):
        print("training started with {} faces, {} non-faces".format(len(faces),len(non_faces)))
        
        img_weights = np.zeros(len(faces)+len(non_faces))
        training_data = []
        
        #print("computing integral images")
        for i in range(len(faces)):
            integral_image = comupute_integral_image(faces[i])
            training_data.append((integral_image, 1))
            img_weights[i] = 1.0 / (2 * len(faces))
        for i in range(len(non_faces)):
            integral_image = comupute_integral_image(non_faces[i])
            training_data.append((integral_image, 0))
            img_weights[len(faces)+i] = 1.0 / (2 * len(non_faces))
        print(len(training_data))
        
        #print("building facial features")
        features = self.build_face_features(training_data[0][0].shape)
        
        #print("applying features...")
        X, y = self.apply_face_features(features, training_data)
        #print("selecting best features...")
        indices = sp(f_classif, percentile=cutoff_percentile).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        #print("selected %d potential features" % len(X))
        
        #evaluate each classifier
        for t in range(self.n_classifiers):
            img_weights = img_weights / np.linalg.norm(img_weights)
            weak_classifiers = self.train_weak(X, y, features, img_weights)
            clf, error, accuracy = self.select_best_weak(weak_classifiers, img_weights, training_data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                img_weights[i] = img_weights[i] * (beta ** (1 - accuracy[i]))
            alpha = np.log(1.0/beta)
            self.learning_rates.append(alpha)
            self.clfs.append(clf)
            
            tot_accuray = np.round(sum(accuracy),2)
            #print("choose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - tot_accuracy, alpha))
           
        
    def train_weak(self, X, y, features, weights):
        #print("in train-weak viola jones")
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            
            clf = weakClassifer(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers
           
    #build features for each img     
    def build_face_features(self, image_shape):
        
        #print(image_shape)
        height, width = image_shape
        features = []
        #for each pixel, create haar features
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        immediate = bounding_box(i, j, w, h)
                        right = bounding_box(i+w, j, w, h)
                        if i + 2 * w < width: 
                            features.append(([right], [immediate]))

                        bottom = bounding_box(i, j+h, w, h)
                        if j + 2 * h < height:
                            features.append(([immediate], [bottom]))
                        
                        right_2 = bounding_box(i+2*w, j, w, h)
                        if i + 3 * w < width: 
                            features.append(([right], [right_2, immediate]))

                        bottom_2 = bounding_box(i, j+2*h, w, h)
                        if j + 3 * h < height: 
                            features.append(([bottom], [bottom_2, immediate]))

                        bottom_right = bounding_box(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features)

    #select best weak classifiers
    def select_best_weak(self, classifiers, weights, training_data):
        #print("selecting best weak classifiers")
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify_weak(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy
    
    #print("classifying image")
    def classify_image(self, img):
        total = 0
        integral = comupute_integral_image(img)
        for alpha, clf in zip(self.learning_rates, self.clfs):
            total += alpha * clf.classify_weak(integral)
        return 1 if total >= 0.5 * sum(self.learning_rates) else 0
    
    #look for facial features
    def apply_face_features(self, features, training_data):
        #print("applying face features")
        i = 0
        img = np.zeros((len(features), len(training_data)))
        label = np.array(list(map(lambda data: data[1], training_data)))
        for pos_box, neg_box in features:
            feature = lambda integral: sum([pos.generate_features(integral) for pos in pos_box]) - sum([neg.generate_features(integral) for neg in neg_box])
            img[i] = list(map(lambda data: feature(data[0]), training_data))
            i += 1
        return img, label

    #load saved Model Class pickle file
    @staticmethod
    def load(filename):
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)
    
    #save Model class object
    def save(self, filename):
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)
    
   
#computes integral image
def comupute_integral_image(img):
    integral = np.zeros(img.shape)
    row_sum = np.zeros(img.shape)
    for j in range(len(img)):
        for i in range(len(img[0])):
            row_sum[j][i] = row_sum[j-1][i] + img[j][i] if j-1 >= 0 else img[j][i]
            integral[j][i] = integral[j][i-1]+row_sum[j][i] if i-1 >= 0 else row_sum[j][i]
    return integral


def downsample_image(img):
    return cv2.resize(img, down_sample_size, interpolation = cv2.INTER_AREA)

def read_folder(folder_path):
    images = []
    for img in glob.glob(folder_path):
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(gray)
    images = np.array(images)
    return images

def make_dataset(faces,non_faces):
    data = []
    for i in range(len(faces)):
        data.append((faces[i], 1))
    for i in range(len(non_faces)):
        data.append((non_faces[i], 0))
    return data

def train_data(t):
    
    faces = read_folder("./code/train/face/*.pgm") #use when running with .sh file
    #faces = read_folder("./test/face/*.pgm")     #use when running with main.py directly
    #faces = faces[:100]
    non_faces = read_folder("./code/train/non-face/*.pgm") #use when running with .sh file
    #non_faces = read_folder("./test/non-face/*.pgm") #use when running with main.py directly
    #non_faces = non_faces[:50]
    print(faces.shape,non_faces.shape)
    
    clf = Model(t)
    train_data = make_dataset(faces, non_faces)
    #train_data = np.random.shuffle(train_data) #shuffling array to avoid any correlation
    clf.train(faces, non_faces)
    fpr,fnr,acc = evaluate_model(clf, train_data)
    clf.save(str(t))
    return fpr,fnr,acc
    
    
def test_data(filename):
    faces = read_folder("./code/test/face/*.pgm") #use when running with .sh file
    #faces = read_folder("./test/face/*.pgm")     #use when running with main.py directly
    #faces = faces[:1000]
    non_faces = read_folder("./code/test/non-face/*.pgm") #use when running with .sh file
    #non_faces = read_folder("./test/non-face/*.pgm")  #use when running with main.py directly 
    #non_faces = non_faces[:50]
    print(faces.shape,non_faces.shape)
    test_data = make_dataset(faces, non_faces)
    #test_data = np.random.shuffle(test_data) #shuffling array to avoid any correlation
    clf = Model.load(filename)
    
    print("testing on {} faces, {} non-faces".format(len(faces),len(non_faces)))
    return evaluate_model(clf, test_data)

def evaluate_model(clf, data):
    #print(len(data))
    #data = np.random.shuffle(data)
    actual,tn, tp, fn, fp = np.zeros(5)
    for x, y in data:
        if y == 1:
            tp += 1
        else:
            tn += 1
        prediction = clf.classify_image(x)
        if prediction == 1 and y == 0:
            fp += 1
        if prediction == 0 and y == 1:
            fn += 1
        actual += 1 if prediction == y else 0
    
    fpr = np.round(fp/tn,2)
    fnr = np.round(fn/tp,2)
    acc = np.round(actual/len(data),2)
    
    print("false-positive count: %d/%d (%.2f)" % (fp, tn, fpr))
    print("false-negative count : %d/%d (%.2f)" % (fn, tp, fnr))
    print("accuracy: %d/%d (%.2f)" % (actual, len(data), acc))
    return fpr,fnr,acc

def plot_results(x,y1,y2,title,labels,fname):
    fig, ax = plt.subplots()#figsize=(10,5))
    plt.plot(x, y1,label=labels[0])
    plt.plot(x, y2,label=labels[1])
    plt.xlabel("classifer count-s")
    plt.ylabel("score")
    plt.title(title)
    plt.legend()
    
    for index in range(len(x)):
      ax.text(x[index], y1[index], y1[index], size=12)
      ax.text(x[index], y2[index], y2[index], size=12)
    plt.savefig("./"+fname+".png")
    #plt.show()
    
    
def main():
    
    #clf_count = 10 #no of weak classifiers to make
    #train_data(clf_count)
    #test_data(str(clf_count))
    
    clf_counts = [5,10,15,20,35,30,35,40]
    fpr_train = []
    fnr_train = []
    acc_train = []
    fpr_test = []
    fnr_test = []
    acc_test = []
    
    for c in clf_counts:
        print("\n\nclassifiers to take: ",c)
        fpr1,fnr1,acc1 = train_data(c)
        fpr2,fnr2,acc2 = test_data(str(c))
        fpr_train.append(fpr1)
        fnr_train.append(fnr1)
        acc_train.append(acc1)
        fpr_test.append(fpr2)
        fnr_test.append(fnr2)
        acc_test.append(acc2)
    
    
    plot_results(clf_counts, fpr_train,fpr_test, "False Posititves", ["train","test"],"FP")
    plot_results(clf_counts, fnr_train,fnr_test, "False Negative", ["train","test"],"FN")
    plot_results(clf_counts, acc_train,acc_test, "Accuracy", ["train","test"],"ACC")
    
main()
    