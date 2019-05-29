<? //https://youtu.be/HA-F6cZPvrg

function sigmoid($x) {return 1.0 / (1.0 + exp(-$x));}
function sigmoid_dx($x) {return ($x * (1.0 - $x));}

$train = [[[0, 0, 0], 0],
          [[0, 0, 1], 1],
          [[0, 1, 0], 0],
          [[0, 1, 1], 0],
          [[1, 0, 0], 1],
          [[1, 0, 1], 1],
          [[1, 1, 0], 0],
          [[1, 1, 1], 0]];

class PartyNN  {
	function __construct($learning_rate=0.1) {
		$this->w[0] = [[.79,.44,.43],[.85,.43,.29]];
		$this->w[1] = [[.5,.52]];
		$this->learning_rate = $learning_rate;
	}
	function predict($inputs) {
													
		$level = 0;											// ¬ход€щие данные, веса текущего сло€, выход€щие данные;
		$vals = $inputs;										// ¬ход - исходные данные 
		$outputs = [];
		do {												// дл€ всех слоЄв...
	
		$weight = $this->w[$level];									// ¬еса; количество = количеству нейронов в сл.слое (!);
		$outputs[$level] = [];										// вџходные значени€; 
		
		for ($j=0; $j < count($weight); $j++) {								// ƒл€ всех вџход€щих

			$dot_sum = 0;										// общий вес
			foreach ($vals as $i=>$a) {								// дл€ каждого входного значени€
				$dot_sum += $vals[$i] * $weight[$j][$i];
			}
			$outputs[$level][$j] = sigmoid($dot_sum);						// ѕосчитать общий вес дл€ одного вых. значени€ (нейрона).
		}
		$vals = $outputs[$level];									// ¬ход - нейроны предыдущего сло€

		} while (++$level < count($this->w));
		
		return $outputs[$level-1];
	}
	function train($inputs, $expected_predict) {

		if (!is_array($expected_predict)) $expected_predict = [$expected_predict];			// ћожет быть много вџходных значений 
													
		$level = 0;											// ¬ход€щие данные, веса текущего сло€, вџход€щие данные;
		$vals = $inputs;										// ¬ход - исходные данные 
		$outputs = [];
		do {												// дл€ всех слоЄв...
	
		$weight = $this->w[$level];									// ¬еса; количество = количеству нейронов в сл.слое (!);
		$outputs[$level] = [];										// вџходные значени€; 

		for ($j=0; $j < count($weight); $j++) {								// ƒл€ всех вџход€щих

			$dot_sum = 0;										// общий вес
			foreach ($vals as $i=>$a) {								// дл€ каждого входного значени€
				$dot_sum += $vals[$i] * $weight[$j][$i];
			}
			$outputs[$level][$j] = sigmoid($dot_sum);						// ѕосчитать общий вес дл€ одного вых. значени€ (нейрона).
		}
		$vals = $outputs[$level];									// ¬ход - нейроны предыдущего сло€

		} while (++$level < count($this->w));

		$error = [];											// ќбратное распространение
		for ($j=0; $j < count($expected_predict); $j++) {						// —читаем ошибку от правильных входных данных ($expected_predict).
			$error[$j] = $outputs[$level-1][$j] - $expected_predict[$j];				// ќшибка дл€ каждого нейрона, j - номер	error - actual - expected
		}
	
		while ($level-- > 0) {										// дл€ всех слоЄв, начина€ слева...

			$actuals = $outputs[$level];								// “екущие результаты, от пр€мого распространени€
			$outputs_prev = $level > 0 ? $outputs[$level-1] : $inputs;				// Ѕерем слой слева; дл€ последнего (левого) сло€ - это вход€щие данные;
			for ($j=0; $j < count($error); $j++) {							// ƒл€ всех вџходных нейронов

				$wdelta = $error[$j] * sigmoid_dx($actuals[$j]);				// дельта весов					weights_delta = error * (sigmoid(x)dx)
				for ($i=0;$i< count($this->w[$level][$j]);$i++) {				// дл€ всех весов нейрона

						$this->w[$level][$j][$i] -=					//						weight_1 = weight_1 - 
							$outputs_prev[$i] * $wdelta * $this->learning_rate;	//						output1 * weights_delta * learning_rate

						$new_error[$i] = $wdelta * $this->w[$level][$j][$i];		// собираем ошибки, дл€ сло€ слева
				}
			}
			$error = $new_error;									// на уровень влево
		}
	}
} // class

$epochs = 1000;
$learning_rate = 0.1;
$network = new PartyNN( $learning_rate );

for ($e = 0; $e < $epochs; $e++) {
    $inputs_ = $correct_predictions = [];
    foreach ($train as $key=>$val) {
	$network->train($val[0],$val[1]);
	$inputs_[] = $val[0];
	$correct_predictions[] = $val[1];
    }
    //$train_loss = MSE($network->mypredict(T($inputs_)), $correct_predictions);
    //printf("\r Progress: %02d,  Training loss: %05f ", ( 100.0 * $e/$epochs), $train_loss);
}

foreach ($train as $key=>$val) {
	printf("\n For input: [%s] the prediction is: [%0.16f], expected: %s ",implode(",",$val[0]), ($network->predict($val[0])[0]), ($val[1]==1?'True':'False'));
}

