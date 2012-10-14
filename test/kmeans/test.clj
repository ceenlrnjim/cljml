(ns kmeans.test
  (:use kmeans)
  (:use clojure.test)
  (:require [incanter.core :as alg]))
 
(deftest test-norm
  (is (= (norm [2 0]) 2.0))
  (is (= (norm [3 4]) 5.0)))

(deftest test-initialize-clusteroids
  (let [ts (alg/matrix [[1.0 1.0][1.0 2.0] [2.0 1.0][2.0 2.0][5.0 1.0][5.0 2.0] [6.0 1.0] [6.0 2.0]])
        c (initialize-clusteroids 2 ts)]
    (is (= (count c) 2))
    (is (every? #(some (partial = %) ts) c))))
    
(deftest test-closest-pt
  (is (= (closest-pt (alg/matrix [[1 1] [100 100]]) (alg/matrix [[2 2]])) 0))
  (is (= (closest-pt (alg/matrix [[1 1] [100 100]]) (alg/matrix [[99 99]])) 1)))

(deftest test-capture-target
  (let [ts [[1 1][1 2] [2 1][2 2][5 1][5 2] [6 1] [6 2]]
        cs [[1.5 1.5] [5.5 1.5]]
        mapping (capture-target ts cs)]
    (is (= mapping {0 [[1 1][1 2][2 1][2 2]] 1 [[5 1][5 2][6 1][6 2]]}))))

;(deftest test-move-clusteroids
      
(defn- test-clusters
  "Takes a sequence of sequences of locations and tests that concatenating them, clustering them returns the same
  grouping"
  [clusters]
  (let [ts (alg/matrix (reduce into [] clusters))
        k (count clusters)
        result (kmeans ts k)
        seterized-clusters (set (map (partial into #{}) clusters)) ; don't want order to matter
        ; TODO: clarify this mess - turning map of sets of matrices into set of set of vectors
        seterized-result (set (map set (map (fn [cluster-set] (map alg/to-vect cluster-set)) (vals result))))]
      (is (= seterized-clusters seterized-result))))

(deftest test-capture-target
  (test-clusters [[[0.0 0.0 1.0] [1.0 5.0 1.0] [10.0 2.0 1.0]] [[0.0 0.0 100.0] [3.0 6.0 100.0] [2.0 8.0 100.0]] [[9.0 7.0 300.0][1.0 0.0 300.0] [2.0 2.0 300.0]]])
  (test-clusters [[[1.0 1.0][1.0 2.0] [2.0 1.0][2.0 2.0]] [[5.0 0.5] [6.0 0.6][5.0 1.0][5.0 2.0] [6.0 1.0] [6.0 2.0]]]))

(run-tests)

