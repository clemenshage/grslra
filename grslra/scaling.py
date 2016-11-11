import numpy as np


class Scaling:
    # This function scales input in a robust way. The input parameters define the target value of a certain magnitude percentile and define whether the data should also be centered before scaling.
    # By default, the data is not centered and is scaled so that the 67th percentile is at 0.3. Thus, Gaussian distributed data is scaled so that almost all data is within the interval [-1, 1]
    def __init__(self, percentile=None, val_at_percentile=None, centering=False):
        self.mu = None
        self.maxval = None
        self.centering = centering
        self.factor = None
        if percentile is not None:
            self.percentile = percentile
        else:
            self.percentile = 67
        if val_at_percentile is not None:
            self.val_at_percentile = val_at_percentile
        else:
            self.val_at_percentile = 0.3

    def scale_reference(self, X, Omega=None, dimensions=None):
        X_scaled = X.copy().astype(np.double)  # copy input, such that the scaling parameters can be determined without actually applying the scaling
        if self.centering:
            # decide whether input is just a vector or a matrix
            if X.shape.__len__() == 1:
                self.mu = np.median(X)
                X_scaled -= self.mu
            else:
                # estimate the median as a column vector across all features
                if Omega is None:
                    self.mu = np.atleast_2d(np.median(X, axis=1)).T
                    X_scaled -= self.mu
                else:
                    if dimensions is not None:
                        m = dimensions[0]
                        self.mu = np.zeros((m, 1))
                        for i in xrange(m):
                            print i
                            ix = np.where(Omega[0] == i)
                            median = np.median(X[ix])
                            X_scaled[ix] -= median
                            self.mu[i, 0] = median
                    else:
                        print "ERROR: Missing dimensions"
                        return X

        self.maxval = np.percentile(np.abs(X_scaled), self.percentile)
        self.factor = self.maxval / self.val_at_percentile
        X_scaled /= self.factor
        return X_scaled

    def scale(self, X, Omega=None, dimensions=None):
        if self.centering:
            if Omega is None:
                X -= self.mu
            else:
                if dimensions is not None:
                    m = dimensions[0]
                    for i in xrange(m):
                        ix = np.where(Omega[0] == i)
                        X[ix] -= self.mu[i, 0]
                else:
                    print "ERROR: Missing dimensions"
                    return X
        X /= self.factor

    def rescale(self, X, Omega=None, dimensions=None):
        X *= self.factor

        if self.centering:
            if Omega is None:
                X += self.mu
            else:
                if dimensions is not None:
                    m = dimensions[0]
                    for i in xrange(m):
                        ix = np.where(Omega[0] == i)
                        X[ix] += self.mu[i, 0]
                else:
                    print "ERROR: Cannot rescale without knowing the dimensions"
