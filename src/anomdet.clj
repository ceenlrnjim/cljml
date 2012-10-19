(ns anomdet
  (:require [incanter.distributions :as dist])
  (:require [incanter.stats :as stats])
  (:use incanter.core))

(defn matrix-means
  "takes an n x m matrix and returns a n dimensional vector 
   containing the mean of each corresponding row from the input"
  [X]
  (let [[rows cols] (dim X)]
    (if (= 1 rows) (stats/mean X) (matrix (map stats/mean X)))))

(defn matrix-variances
  "takes an n x m matrix of examples and a nx1 matrix of the means of those samples and returns a n dimensional vector 
   containing the variance of each corresponding row from the input"
  [X means]
  {:pre [(matrix? X) (= (first (dim X)) (first (dim means))) (= 1 (second (dim means)))]
  :port [(or (number? %) (= (second (dim %)) 1))]}
                     ;TODO: using this instead of variance to prevent having to recompute a mean I already have
                     ; Need to validate it is actually faster
    (matrix (map #(/ (stats/sum-of-square-devs-from-mean %1 %2) (count %1)) X means)))


; more computationally expensive
(comment
(defn multivar-density
  "nx1 matrix for xi and means, nxn for sigma"
  [xi covar means]
  (/ (exp (mmult (mult -1.5 (trans (minus xi means))) (solve covar) (minus x means)))
    (* (Math/pow (* 2 Math/PI) (/ (count xi) 2)) (Math/pow (det covar) 0.5))))
)

;univariate guassian distribution - faster to compute
(defn multi-pdf
  "example is an n-dimensional feature vector for one example (nx1), 
   means is an n-dimensional vector of the computed means for the training set,
   vars is an n-dimensional vector of the variances for the training set"
  [example means vars]
  {:pre [(= (first (dim example)) (first (dim means)) (first (dim vars)))]
   :post [(number? %)]}
  (reduce
    *
    (map #(dist/pdf (dist/normal-distribution %1 (Math/sqrt %2)) %3) means vars example)))

(defn anomaly-detector
  "Returns a function that returns true/false for an example with same features as
  the specified training set indicating whether or not it is anomalous. 
  trainset is a mxn matrix (#example by #features)"
  [trainset threshold]
  (let [means (matrix-means (trans trainset))
        vars (matrix-variances (trans trainset) means)]
    (fn [example]
      ; application for reducers?
      (let [pct (multi-pdf example means vars)]
        (println "found pct " pct " for example " example)
        (< pct threshold)))))
