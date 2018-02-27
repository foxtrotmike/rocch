from scipy.spatial import ConvexHull
import numpy as np
def rocch(fpr0,tpr0):
    """
    @author: Dr. Fayyaz Minhas (http://faculty.pieas.edu.pk/fayyaz/)
    Construct the convex hull of a Receiver Operating Characteristic (ROC) curve
        Input:
            fpr0: List of false positive rates in range [0,1]
            tpr0: List of true positive rates in range [0,1]
                fpr0,tpr0 can be obtained from sklearn.metrics.roc_curve or 
                    any other packages such as pyml
        Return:
            F: list of false positive rates on the convex hull
            T: list of true positive rates on the convex hull
                plt.plot(F,T) will plot the convex hull
            auc: Area under the ROC Convex hull
    """
    fpr = np.array([0]+list(fpr0)+[1.0,1,0])
    tpr = np.array([0]+list(tpr0)+[1.0,0,0])
    hull = ConvexHull(np.vstack((fpr,tpr)).T)
    vert = hull.vertices
    vert = vert[np.argsort(fpr[vert])]  
    F = [0]
    T = [0]
    for v in vert:
        ft = (fpr[v],tpr[v])
        if ft==(0,0) or ft==(1,1) or ft==(1,0):
            continue
        F+=[fpr[v]]
        T+=[tpr[v]]
    F+=[1]
    T+=[1]
    auc = np.trapz(T,F)
    return F,T,auc
