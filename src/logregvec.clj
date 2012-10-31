(ns logregvec
  (:require [incanter.core :as alg]))


; Utility functions -----------------------------------------------
(defn featurecount [X]
  (count (first X)))

(defn uniform-vector [size val]
  (alg/matrix (map (fn [_] (vector val)) (range 0 size))))

(defn empty-vector [size]
  (uniform-vector size 0))

; Calculations functions -----------------------------------------------

(defn hyperror [hypFn X yvec]
  (alg/minus (hypFn X) yvec))

(defn dJdtheta [hypFn X yvec]
  (alg/mmult (alg/trans X) (hyperror hypFn X yvec)))

(defn reductionamt [hypFn X yvec alpha]
  (alg/mult (/ alpha (count yvec)) (dJdtheta hypFn X yvec)))

(defn next-thetas
  [hypothesis thetas alpha X yvec]
  (alg/minus thetas (reductionamt (partial hypothesis thetas) X yvec alpha)))


; API functions -----------------------------------------------
(defn optimize [X y alpha maxiters hypothesis]
  (loop [iters maxiters
       thetas (empty-vector (featurecount X))]
    (if (= 0 iters) thetas
      (recur 
        (dec iters)
        (next-thetas hypothesis thetas alpha X y)))))


; Algorithm functions -----------------------------------------------
(defn sigmoid [X]
  "Returns the value of the sigmoid function for each value in the matrix X"
  (alg/matrix-map #(/ 1 (+ 1 (alg/exp (* -1 %)))) X))

(defn logistic-hypothesis [thetas X]
  "Given parameters thetas, returns the predicted value for all the inputs in the training set matrix X"
  (sigmoid (alg/mmult X thetas)))

(defn linear-hypothesis [thetas X]
  (alg/mmult X thetas))

; TODO: move all this stuff out and make unit tests
; Execute a logistic test
(def myX (alg/matrix [[1 0 0] [1 0 1] [1 1.5 1] [1 1.5 2] [1 0 2] [1 0 3] [1 1 0] [1 1 1] [1 1 2] [1 1 3] [1 2 0] [1 2 1] [1 2 2] [1 2 3]]))
(def myy (alg/matrix [[0] [0] [0] [0] [0] [0] [0] [0] [0] [0] [1] [1] [1] [1]]))
(def opt-thetas (time (optimize myX myy 0.9 10000 logistic-hypothesis)))
(println opt-thetas)
(println "probability of 1 for 0,0" (logistic-hypothesis opt-thetas (alg/matrix [[1 0 0]])))
(println "probability of 1 for 1,0" (logistic-hypothesis opt-thetas (alg/matrix [[1 1 0]])))
(println "probability of 1 for 1.5,0" (logistic-hypothesis opt-thetas (alg/matrix [[1 1.5 0]])))
(println "probability of 1 for 2,0" (logistic-hypothesis opt-thetas (alg/matrix [[1 2 0]])))
(println "probability of 1 for 3,0" (logistic-hypothesis opt-thetas (alg/matrix [[1 3 0]])))
(println "probability of 1 for 4,0" (logistic-hypothesis opt-thetas (alg/matrix [[1 4 0]])))
(println "probability of 1 for 5,0" (logistic-hypothesis opt-thetas (alg/matrix [[1 5 0]])))
(comment
)

; Execute a linear test
(def linX (alg/matrix [[1 0] [1 1] [1 2] [1 3]]))
(def linY (alg/matrix [0 1 2 3]))
(def opt-thetas (optimize linX linY 0.09 2000 linear-hypothesis))
(println opt-thetas)
