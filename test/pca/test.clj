(ns pca.test
  (:use pca)
  (:use clojure.test)
  (:require [incanter.core :as alg]))

; TODO: more tests for this
(deftest test-mean-normalize
  (let [X (alg/matrix [[1 1 1][2 2 2][3 3 3]])
        mn (mean-normalize X)]
    (is (= mn (alg/matrix [[-1.0 -1.0 -1.0][0.0 0.0 0.0][1.0 1.0 1.0]])))))

(deftest test-reduce-dimensions
  (let [X (mean-normalize (alg/matrix [[1 1 1 2 2 2][2 2 2 3 3 3][3 3 3 4 4 4]]))
        result (reduce-dimensions X 2)]
        ))
