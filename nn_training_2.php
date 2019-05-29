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
													
		$level = 0;											// Входящие данные, веса текущего слоя, выходящие данные;
		$vals = $inputs;										// Вход - исходные данные 
		$outputs = [];
		do {												// для всех слоёв...
	
		$weight = $this->w[$level];									// Веса; количество = количеству нейронов в сл.слое (!);
		$outputs[$level] = [];										// вЫходные значения; 
		
		for ($j=0; $j < count($weight); $j++) {								// Для всех вЫходящих

			$dot_sum = 0;										// общий вес
			foreach ($vals as $i=>$a) {								// для каждого входного значения
				$dot_sum += $vals[$i] * $weight[$j][$i];
			}
			$outputs[$level][$j] = sigmoid($dot_sum);						// Посчитать общий вес для одного вых. значения (нейрона).
		}
		$vals = $outputs[$level];									// Вход - нейроны предыдущего слоя

		} while (++$level < count($this->w));
		
		return $outputs[$level-1];
	}
	function train($inputs, $expected_predict) {

		if (!is_array($expected_predict)) $expected_predict = [$expected_predict];			// Может быть много вЫходных значений 
													
		$level = 0;											// Входящие данные, веса текущего слоя, вЫходящие данные;
		$vals = $inputs;										// Вход - исходные данные 
		$outputs = [];
		do {												// для всех слоёв...
	
		$weight = $this->w[$level];									// Веса; количество = количеству нейронов в сл.слое (!);
		$outputs[$level] = [];										// вЫходные значения; 

		for ($j=0; $j < count($weight); $j++) {								// Для всех вЫходящих

			$dot_sum = 0;										// общий вес
			foreach ($vals as $i=>$a) {								// для каждого входного значения
				$dot_sum += $vals[$i] * $weight[$j][$i];
			}
			$outputs[$level][$j] = sigmoid($dot_sum);						// Посчитать общий вес для одного вых. значения (нейрона).
		}
		$vals = $outputs[$level];									// Вход - нейроны предыдущего слоя

		} while (++$level < count($this->w));

		$error = [];											// Обратное распространение
		for ($j=0; $j < count($expected_predict); $j++) {						// Считаем ошибку от правильных входных данных ($expected_predict).
			$error[$j] = $outputs[$level-1][$j] - $expected_predict[$j];				// Ошибка для каждого нейрона, j - номер	error - actual - expected
		}
	
		while ($level-- > 0) {										// для всех слоёв, начиная слева...

			$actuals = $outputs[$level];								// Текущие результаты, от прямого распространения
			$outputs_prev = $level > 0 ? $outputs[$level-1] : $inputs;				// Берем слой слева; для последнего (левого) слоя - это входящие данные;
			for ($j=0; $j < count($error); $j++) {							// Для всех вЫходных нейронов

				$wdelta = $error[$j] * sigmoid_dx($actuals[$j]);				// дельта весов					weights_delta = error * (sigmoid(x)dx)
				for ($i=0;$i< count($this->w[$level][$j]);$i++) {				// для всех весов нейрона

						$this->w[$level][$j][$i] -=					//						weight_1 = weight_1 - 
							$outputs_prev[$i] * $wdelta * $this->learning_rate;	//						output1 * weights_delta * learning_rate

						$new_error[$i] = $wdelta * $this->w[$level][$j][$i];		// собираем ошибки, для слоя слева
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

