(ns nn.test
    (:use nn)
    (:require [incanter.core :as alg])
    (:use clojure.test))


(defn- closeto?
  ([x y] (closeto? x y 0.001))
  ([x y tolerance] (< (Math/abs (- y x)) tolerance)))

(deftest test-sigmoid
  (is (= (sigmoid 0) 0.5))
  (is (< (sigmoid -10) 0.0001))
  (is (> (sigmoid 10) 0.9999))
  (let [x (alg/matrix [[1 2]])
        gx (sigmoid x)]
    (is (= (alg/dim x) (alg/dim gx))))
  (let [m (alg/matrix [[-10 0 10] [-10 0 10] [-10 0 10]])
        sigm (sigmoid m)]
    (is (< (alg/$ 0 0 sigm) 0.0001))
    (is (= (alg/$ 0 1 sigm) 0.5))
    (is (> (alg/$ 0 2 sigm) 0.9999))))

(deftest test-sigmoid-gradient
  (let [m (alg/matrix [[1 2 3] [4 5 6]])
        sgm (sigmoid-gradient m)
        octave-answer (alg/matrix [[0.1966119 0.1049936 0.0451767][0.0176627 0.0066481 0.0024665]])]
    (is (every? (partial closeto? 0) (alg/vectorize (alg/minus octave-answer sgm))))))

(deftest test-drop-column
  (let [m (alg/matrix [[1 2 3][1 2 3][1 2 3]])]
    (is (= (drop-column m 0) (alg/matrix [[2 3][2 3][2 3]])))
    (is (= (drop-column m 1) (alg/matrix [[1 3][1 3][1 3]])))
    (is (= (drop-column m 2) (alg/matrix [[1 2][1 2][1 2]])))))

(deftest test-rand-theta
  (let [m (rand-theta 3 2)]
    (is (alg/matrix? m))
    (is (= (alg/dim m) [3 2]))
    (is (not (every? (partial = 0) m)))))

(deftest test-initialize-parameters
  (let [m (initialize-parameters [5 10 10 1])] 
    (is (= (count m) 3))
    (is (= (alg/dim (m 1)) [10 6]))
    (is (= (alg/dim (m 2)) [10 11]))
    (is (= (alg/dim (m 3)) [1 11]))))

(deftest test-column-cat
  ; test the multi-dimensional matrix case
  (println "testing 4x2 and 4x2")
  (is (= (alg/matrix [[1 2 3 4] [1 2 3 4] [1 2 3 4][1 2 3 4]])
        (column-cat (alg/matrix [[1 2][1 2][1 2][1 2]])
                    (alg/matrix [[3 4][3 4][3 4][3 4]]))))
  ; test two single dimensional vertical vectors
  (println "testing 4x1 and 4x1")
  (is (= (alg/matrix [[1 2][1 2][1 2][1 2]]) 
        (column-cat (alg/matrix [[1][1][1][1]]) (alg/matrix [[2][2][2][2]]))))
  ; test for the pain-in-the-ass matrix to vector conversion stuff
  (println "testing 1x2 and 1x1")
  (let [x (alg/matrix [[1.0 1.0]])
        y (alg/matrix [[1.0]])
        result (column-cat x y)]
    (is (=  result (alg/matrix [[1.0 1.0 1.0]])))
    (is (= (alg/dim result) [1 3])))
  ; test for more single dimension stuff
  (println "testing 1x3 vector and 3x1 matrix")
  (is (= (alg/matrix [[1 2] [1 2] [1 2]])
        (column-cat (alg/matrix [1 1 1]) (alg/matrix [[2][2][2]])))))

(deftest test-forward-prop
  ; using a XNOR logic "gate" to test since I can pick the theta values manually
  (let [X (alg/matrix [[0 0]])
        thetas {1 (alg/matrix [[-30 20 20][10 -20 -20]])
                2 (alg/matrix [[-10 20 20]])}
        prediction (:prediction (forward-prop X thetas))]
    (is (every? #(closeto? 0 %) (alg/minus (alg/matrix [[1.0]]) prediction))))
  (let [X (alg/matrix [[0 0][0 1] [1 0][1 1]])
        thetas {1 (alg/matrix [[-30 20 20][10 -20 -20]])
                2 (alg/matrix [[-10 20 20]])}
        prediction (:prediction (forward-prop X thetas))]
    (is (every? #(closeto? 0 %) (alg/minus (alg/matrix [[1.0][0.0][0.0][1.0]]) prediction)))))

(deftest test-regularize
  (let [thetas {1 (alg/matrix [[2 2][2 2]]) 2 (alg/matrix [3 3]) 3 (alg/matrix [[3 3]])}
        sumsquared (+ 16 18 18)]
    (is (= (regularize 0 100 thetas) 0.0))
    (is (= (regularize 1 100 thetas) (double (/ sumsquared 200))))))
  

; TODO: can't have a 0 in prediction or log goes to -infinity
(deftest test-example-cost
  ; answers come from octave
  (is (closeto? -0.030151 (example-cost (alg/matrix [[0.99 0.01 0.01]]) (alg/matrix [[1 0 0]]))))
  (is (closeto? -0.16252 (example-cost (alg/matrix [[0.85]]) (alg/matrix [[1]]))))
  (is (closeto? -2.2493 (example-cost (alg/matrix [[0.25 0.25 0.25 0.25]]) (alg/matrix [[0 0 1 0]]))))
  (let [prediction (:prediction (forward-prop
                                  (alg/matrix [[0 0]])
                                  ; xnor network
                                  {1 (alg/matrix [[-30 20 20][10 -20 -20]])
                                   2 (alg/matrix [[-10 20 20]])}))
        actual 1.0]
    (is (closeto? 0.0 (example-cost prediction actual)))))
              

(deftest test-cost
  (let [X (alg/matrix [[0 0][0 1] [1 0][1 1]])
        thetas {1 (alg/matrix [[-30 20 20][10 -20 -20]])
                2 (alg/matrix [[-10 20 20]])}
        xnorcost (cost thetas X (alg/matrix [[1.0][0.0][0.0][1.0]]) 0.0)]
    (is (closeto? 0.0 xnorcost))
    ))

(deftest test-network-errors
  (let [X (alg/matrix [[0 0][0 1] [1 0][1 1]])
        thetas {1 (alg/matrix [[-30 20 20][10 -20 -20]])
                2 (alg/matrix [[-10 20 20]])}
        expected (alg/matrix [[1.0][0.0][0.0][1.0]])
        fpr (forward-prop X thetas)
        errors (network-errors fpr thetas expected)]
    ;TODO: need more test cases and tests
    (is (= (count errors) 2)) ; no errors for the input layer
    (println errors)
    ; passing in the right answer so errors should be very small
    (is (every? #(closeto? 0 %) (errors 3)))
    (is (every? #(closeto? 0 %) (alg/vectorize (errors 2))))))

(deftest test-gradients
  (println "test-gradients > ------------------------------------------")
         ;TODO: change to use the new unrolled syntax
  (let [X (alg/matrix [[0 0][0 1] [1 0][1 1]])
        thetas {1 (alg/matrix [[-5 5 1][10 -20 -20]])
                2 (alg/matrix [[-10 20 20]])}
        expected (alg/matrix [[1.0][0.0][0.0][1.0]])
        grads (gradients X thetas expected 0)]
    ; TODO: implement some tests
    (println "gradients:" grads)))

(deftest test-unroll
  (let [x (alg/matrix [[1 1 1] [2 2 2] [3 3 3]])
        y (alg/matrix [[4 4][5 5][6 6][7 7][8 8]])
        result (unroll {1 x 2 y})]
    (is (= result (alg/matrix [1.0 2.0 3.0 1.0 2.0 3.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 4.0 5.0 6.0 7.0 8.0])))))

(deftest test-reroll
         ; 2 inputs, 3 hidden units, 1 output
  (let [x1-unrolled (alg/matrix [[1 1 1 1 1 1 1 1 1 2 2 2 2]])
        x1 {1 (alg/matrix [[1 1 1][1 1 1][1 1 1]])
            2 (alg/matrix [[2 2 2 2]])}]
    (is (= (reroll x1-unrolled [2 3 1]) x1))))

(deftest test-layer-dims
  (is (= (layer-dims [2 4 4 1])
         [[4 3][4 5][1 5]])))
