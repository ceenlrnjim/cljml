(ns logregvec
  (:require [incanter.core :as alg]))


; Calculations functions -----------------------------------------------
(defn next-thetas
  "Computes the next value of theta given the gradients and current theta values"
  [thetas learning-rate gradients]
  (alg/minus 
    thetas 
    (alg/mult learning-rate (gradients thetas))))


; API functions -----------------------------------------------
; TODO: add regularization support
; TODO: where do I need the cost function? Just to track progress?
; TODO: change to map based arguments for clarity?
(defn optimize [initial-thetas learning-rate costfn gradientfn maxiters]
  (loop [iters maxiters
         thetas initial-thetas]
    ;(println "Iteration: " iters " cost=" (costfn thetas))
    (if (= 0 iters) thetas
      (recur 
        (dec iters)
        (next-thetas thetas learning-rate gradientfn)))))


; Algorithm functions -----------------------------------------------
(defn sigmoid
  "Returns the value of the sigmoid function for each value in the matrix X"
  [X]
  (alg/matrix-map #(/ 1 (+ 1 (alg/exp (* -1 %)))) X))

(defn logistic-hypothesis 
  "Given parameters thetas, returns the predicted value for all the inputs in the training set matrix X"
  [thetas X]
  (sigmoid (alg/mmult X thetas)))

(defn linear-hypothesis
  "Given parameters thetas, returns the predicted value for all the inputs in the training set matrix X which is
   expected to have a row for each example with a column for each feature"
  [thetas X]
  (alg/mmult X thetas))

(defn linlog-cost
  "returns the cost for the specified thetas and hypothsis function using the squared error"
  [hypfn X y]
  (fn [thetas]
    (let [[m n] (alg/dim X)]
      (alg/mult (/ 1 (* 2 m)) (alg/pow (alg/minus (hypfn thetas X) y))))))

(defn linlog-gradients 
  "Returns a function of parameters theta that return the gradient value relative to each feature xj"
  [hypfn X y]
  (fn [thetas]
    (let [[setsize features] (alg/dim X)
          predicted (hypfn thetas X)
          errors (alg/minus predicted y)]
      (alg/mult (/ 1 setsize) (alg/mmult (alg/trans X) errors)))))
