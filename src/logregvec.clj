(ns logregvec
  (:require [incanter.core :as alg]))


; Utility functions -----------------------------------------------
; TODO: move to utility package - check for duplicates in other libraries
(defn feature-count [X]
  {:pre [(alg/matrix? X)] }
  (second (alg/dim X)))

(defn uniform-vector [size val]
  (alg/matrix (map (fn [_] (vector val)) (range 0 size))))

(defn empty-vector [size]
  (uniform-vector size 0))

; Calculations functions -----------------------------------------------

(defn hyperror [hypFn X yvec]
  (alg/minus (hypFn X) yvec))

; TODO: change gradients to be arguments (as in fminunc etc.) to support standard interface for minimization
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
       thetas (empty-vector (feature-count X))]
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
