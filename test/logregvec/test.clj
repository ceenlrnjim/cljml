(ns logregvec.test
(:use logregvec)
(:use clojure.test)
(:require [incanter.core :as alg]))


; TODO: move to test utils
(defn- closeto?
  ([x y] (closeto? x y 0.001))
  ([x y tolerance] (< (Math/abs (- y x)) tolerance)))

(deftest test-optimize-logistic
  (let [X (alg/matrix [[1 0 0] [1 0 1] [1 1.5 1] [1 1.5 2] [1 0 2] [1 0 3] [1 1 0] [1 1 1] [1 1 2] [1 1 3] [1 2 0] [1 2 1] [1 2 2] [1 2 3]])
        y (alg/matrix [[0] [0] [0] [0] [0] [0] [0] [0] [0] [0] [1] [1] [1] [1]])
        opt-thetas (optimize {:initial-thetas (alg/matrix [0 0 0])
                              :learning-rate 0.9 
                              :max-iters 10000 
                              :cost-fn (linlog-cost logistic-hypothesis X y)
                              :gradient-fn (linlog-gradients logistic-hypothesis X y)})]
    (is (closeto? 0 (logistic-hypothesis opt-thetas (alg/matrix [[1 0 0]])))) ; make predictions with optimal thetas to test
    (is (closeto? 1 (logistic-hypothesis opt-thetas (alg/matrix [[1 5 0]]))))))

(deftest test-optimize-linear
  (let [X (alg/matrix [[1 0] [1 1] [1 2] [1 3]])
        y (alg/matrix [0 1 2 3])
        opt-thetas (optimize {:initial-thetas (alg/matrix [0 0]) 
                              :learning-rate 0.09 
                              :max-iters 2000 
                              :cost-fn (linlog-cost linear-hypothesis X y)
                              :gradient-fn (linlog-gradients linear-hypothesis X y)})]
    (is (closeto? 4 (linear-hypothesis opt-thetas (alg/matrix [[1 4]]))))
    (is (closeto? 100 (linear-hypothesis opt-thetas (alg/matrix [[1 100]]))))))
