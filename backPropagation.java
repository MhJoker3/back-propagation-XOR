package hw5;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class backPropagation {

	private List<ArrayList<Double>> inW;
	private List<ArrayList<Double>> hideW;
	private List<ArrayList<Double>> trainSet;

	double rate;
	double error;
	double input = 2;
	double hidden = 2;
	double output = 1;

	/**
	 * construct function
	 * 
	 * @param trainSet
	 * @param error
	 * @param rate
	 */
	public backPropagation(List<ArrayList<Double>> trainSet, double error, double rate) {
		this.trainSet = trainSet;
		this.error = error;
		this.rate = rate;
	}

	/**
	 * train the dataset
	 */
	public void train() {

		// initialize needed data structure
		List<Double> hideValue = new ArrayList<>();
		for (int i = 0; i < hidden; i++) {
			hideValue.add(0.0);
		}

		List<Double> outValue = new ArrayList<>();
		for (int i = 0; i < output; i++) {
			outValue.add(0.0);
		}

		List<Double> inout = new ArrayList<>();
		for (int i = 0; i < hidden; i++) {
			inout.add(0.0);
		}

		List<Double> hideout = new ArrayList<>();
		for (int i = 0; i < output; i++) {
			hideout.add(0.0);
		}

		List<Double> hideErr = new ArrayList<>();
		for (int i = 0; i < hidden; i++) {
			hideErr.add(0.0);
		}

		double e = 0.0;

		List<ArrayList<Double>> deltainW = new ArrayList<ArrayList<Double>>();
		List<ArrayList<Double>> deltahideW = new ArrayList<ArrayList<Double>>();

		// initial deltainW to 0
		for (int i = 0; i < inW.size(); i++) {
			ArrayList<Double> tmp = new ArrayList<>();
			for (int j = 0; j < inW.get(i).size(); j++) {
				tmp.add(0.0);
			}
			deltainW.add(i, tmp);
		}

		// initial deltahideW to 0
		for (int i = 0; i < hideW.size(); i++) {
			ArrayList<Double> tmp = new ArrayList<>();
			for (int j = 0; j < hideW.get(i).size(); j++) {
				tmp.add(0.0);
			}
			deltahideW.add(i, tmp);
		}

		// deal with each four dataset
		for (int index = 0; index < trainSet.size(); index++) {

			// calculate hide value
			for (int i = 0; i < hidden; i++) {
				Double tmp = 0.0;
				for (int j = 0; j < input; j++) {
					tmp += trainSet.get(index).get(j) * inW.get(j).get(i);
				}
				inout.set(i, tmp);
				hideValue.set(i, activateFunc(inout.get(i)));
			}

			// calculate output value
			for (int i = 0; i < output; i++) {
				Double tmp = 0.0;
				for (int j = 0; j < hidden; j++) {
					tmp += hideValue.get(j) * hideW.get(j).get(i);
				}
				hideout.set(i, tmp);
				outValue.set(i, activateFunc(hideout.get(i)));
			}

			// calculate error for each dataset
			double err = (trainSet.get(index).get(2) - outValue.get(0)) * (1 - outValue.get(0)) * outValue.get(0);
			double er = trainSet.get(index).get(2) - outValue.get(0);
			e += Math.abs(er) * Math.abs(er);

			// calculate delta hide weight
			for (int i = 0; i < hidden; i++) {
				ArrayList<Double> tmp1 = new ArrayList<>();
				Double temp = rate * err * hideValue.get(i);
				tmp1.add(temp + deltahideW.get(i).get(0));
				deltahideW.set(i, tmp1);
			}

			// calculate hide error
			for (int i = 0; i < hidden; i++) {
				hideErr.set(i, err * hideW.get(i).get(0) * (1 - hideValue.get(i)) * hideValue.get(i));
			}

			// calculate delta input weight
			for (int i = 0; i < input; i++) {
				ArrayList<Double> tmp2 = new ArrayList<>();
				for (int j = 0; j < hidden; j++) {
					Double temp1 = rate * hideErr.get(j) * trainSet.get(index).get(i);
					// System.out.println("....."+deltainW.size()+","+deltainW.get(i).size());
					tmp2.add(j, temp1 + deltainW.get(i).get(j));
				}
				deltainW.set(i, tmp2);
			}
		}

		// add delta input weight to input weight
		for (int i = 0; i < inW.size(); i++) {
			ArrayList<Double> tmp2 = new ArrayList<>();
			for (int j = 0; j < inW.get(0).size(); j++) {
				Double temp2 = 0.0;
				temp2 += inW.get(i).get(j) + deltainW.get(i).get(j);
				tmp2.add(temp2);
			}
			inW.set(i, tmp2);
		}

		// add delta hide weight to hide weight
		for (int i = 0; i < hideW.size(); i++) {
			ArrayList<Double> tmp3 = new ArrayList<>();
			for (int j = 0; j < hideW.get(0).size(); j++) {
				Double temp3 = 0.0;
				temp3 += hideW.get(i).get(j) + deltahideW.get(i).get(j);
				tmp3.add(temp3);
			}
			hideW.set(i, tmp3);
		}

		// calculate error
		error = e / 8;
	}

	/**
	 * initialize input weight and hide weight to -1 ~ 1
	 */
	public void init() {

		inW = new ArrayList<ArrayList<Double>>();
		hideW = new ArrayList<ArrayList<Double>>();

		int min = -1;
		int max = 1;

		for (int i = 0; i < input; i++) {
			ArrayList<Double> tmp = new ArrayList<Double>();
			for (int j = 0; j < hidden; j++) {
				tmp.add((min + (Math.random() * (max - min))));
			}
			inW.add(tmp);
		}

		for (int i = 0; i < hidden; i++) {
			ArrayList<Double> tmp = new ArrayList<Double>();
			for (int j = 0; j < output; j++) {
				tmp.add((min + (Math.random() * (max - min))));
			}
			hideW.add(tmp);
		}
	}

	/**
	 * activate function
	 * 
	 * @param x
	 * @return
	 */
	private double activateFunc(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	/**
	 * get input weight
	 * 
	 * @return
	 */
	public List<ArrayList<Double>> getinW() {
		return inW;
	}

	/**
	 * get hide weight
	 * 
	 * @return
	 */
	public List<ArrayList<Double>> gethideW() {
		return hideW;
	}
}
