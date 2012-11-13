(ns logregvec
  (:require [incanter.core :as alg]))

; helper functions
; TODO: must be a better way
(defn set-vector-value-at
  "returns the specified vector with the specified value at the specified index"
  [v ix value]
  (let [multiplier (map-indexed (fn [i _] (if (= i ix) 0 1)) v)
        adder (map-indexed (fn [i _] (if (= i ix) value 0)) v)]
    (alg/plus adder (alg/mult multiplier v))))

; Calculations functions -----------------------------------------------
(defn next-thetas
  "Computes the next value of theta given the gradients and current theta values"
  [thetas learning-rate gradients]
  (alg/minus 
    thetas 
    (alg/mult learning-rate (gradients thetas))))


; API functions -----------------------------------------------
; TODO: where do I need the cost function? Just to track progress?
(defn optimize [{:keys [initial-thetas learning-rate cost-fn gradient-fn max-iters]}]
  (loop [iters max-iters
         thetas initial-thetas]
    ;(println "Iteration: " iters " cost=" (cost-fn thetas))
    (if (= 0 iters) thetas
      (recur 
        (dec iters)
        (next-thetas thetas learning-rate gradient-fn)))))

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
  ([hypfn X y] (linlog-cost hypfn X y 0))
  ([hypfn X y regparam]
  (fn [thetas]
    (let [[m n] (alg/dim X)
          regterm (* regparam (alg/sum (alg/pow thetas 2)))] ; TODO: theta0 regularized as well?
      ; TODO: is this 'sum' correct - I don't want a vector of costs, but a scalar - double check with notes
      (* (/ 1 (* 2 m))
        (+ regterm 
          (alg/sum 
            (alg/pow 
              (alg/minus (hypfn thetas X) y) 
              2))))))))

(defn linlog-gradients 
  "Returns a function of parameters theta that return the gradient value relative to each feature xj"
  ([hypfn X y] (linlog-gradients hypfn X y 0))
  ([hypfn X y regparam]
    (fn [thetas]
      (let [[setsize features] (alg/dim X)
            predicted (hypfn thetas X)
            errors (alg/minus predicted y)
            regterm (set-vector-value-at (alg/mult (/ regparam setsize) thetas) 0 0)]
        ; Don't want to regularize x0 term (theta j where j=0)
        (alg/plus regterm (alg/mult (/ 1 setsize) (alg/mmult (alg/trans X) errors)))))))
