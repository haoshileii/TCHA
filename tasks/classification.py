import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from . import drawsvm as draw
from models.lossestriples import hierarchical_contrastive_loss
# import matplotlib
# matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as ply
def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    train_repr, trainlabel_repr = model.encode(train_data, train_labels, encoding_window='full_series' if train_labels.ndim == 1 else None)
    #np.save('E:/code/HCL/datasets/UEAHCL/' + dataname + '_TRAIN2', train_repr)
    #np.save('E:/code/HCL/datasets/UEAHCL/' + dataname + '_TRAIN_LABEL2', trainlabel_repr)
    test_repr, testlabel_repr= model.encode(test_data, test_labels, encoding_window='full_series' if train_labels.ndim == 1 else None)
    #np.save('E:/code/HCL/datasets/UEAHCL/' + dataname + '_TEST2', test_repr)
    #np.save('E:/code/HCL/datasets/UEAHCL/' + dataname + '_TEST_LABEL2', testlabel_repr)
   # draw.draw_tsne_2D(test_data, test_labels, 1)
    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])
    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)
    #draw.draw_tsne_2D(test_repr, test_labels, 2)
    clf = fit_clf(train_repr, train_labels)
    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    return y_score, { 'acc': acc, 'auprc': auprc }
