(ns anomdet.test
  (:use anomdet)
  (:use clojure.test)
  (:require [incanter.stats :as stats])
  (:require [incanter.distributions :as dist])
  (:use incanter.core))


(defn- closeto? 
  ([x y] (closeto? x y 0.001))
  ([x y tolerance] (< (Math/abs (- y x)) tolerance)))

(deftest test-matrix-mean
  (is (= (matrix-means (matrix [[2 2 2 2]])) 2.0))
  (is (= (matrix-means (matrix [[1 2 4 5][7 8 10 11]])) (matrix [[3.0][9.0]]))))

(deftest test-matrix-variances
          ; vector of 4 normally distributed sequences with stddev 
  (let [samples (map #(stats/sample-normal 10000 :mean 0 :sd %) [1 2 3 4])
        computed-vars (matrix-variances (matrix samples) (matrix-means samples))]
    (println computed-vars)
    (is (every? true? (map #(closeto?%1 %2 0.5) [1 4 9 16] computed-vars))))) ; variance here, std dev above

(deftest test-anomaly-detector
                  ; 10,000 examples x 2 features
  (let [trainset (trans (matrix [(stats/sample-normal 10000 :mean 0 :sd 1)
                          (stats/sample-normal 10000 :mean 5 :sd 1)]))
        anomalous? (anomaly-detector trainset 0.02)]
    (is (not (anomalous? (matrix [1.05 5.05]))))
    (is (anomalous? (matrix [5 1])))
    (is (anomalous? (matrix [50 50])))
    (is (anomalous? (matrix [50 5])))
    (is (anomalous? (matrix [1 50])))))
