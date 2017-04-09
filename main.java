package hw5;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class main {
	public static void main(String[] args) {

		// initialize trainset
		List<ArrayList<Double>> trainSet = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> tmp1 = new ArrayList<>();
		tmp1.add(0.0);
		tmp1.add(0.0);
		tmp1.add(0.0);
		trainSet.add(tmp1);
		ArrayList<Double> tmp2 = new ArrayList<>();
		tmp2.add(0.0);
		tmp2.add(1.0);
		tmp2.add(1.0);
		trainSet.add(tmp2);
		ArrayList<Double> tmp3 = new ArrayList<>();
		tmp3.add(1.0);
		tmp3.add(0.0);
		tmp3.add(1.0);
		trainSet.add(tmp3);
		ArrayList<Double> tmp4 = new ArrayList<>();
		tmp4.add(1.0);
		tmp4.add(1.0);
		tmp4.add(0.0);
		trainSet.add(tmp4);

//		String choice = "Y";
		while (true) {

			// input learning rate and expected error
			double rate;
			double error;
			System.out.println("Please input learning rate:");
			Scanner sc = new Scanner(System.in);
			rate = sc.nextDouble();
			System.out.println("Please input expected error:");
			error = sc.nextDouble();

			// create backPropagation instance
			backPropagation bp = new backPropagation(trainSet, error, rate);
			bp.init();

			int i = 0;
			double firstErr = 0.0;
			double finalErr = 0.0;
			int numBatch = 0;

			List<ArrayList<Double>> initinW = bp.getinW();

			System.out.println("the initial input weight---------------");
			print(initinW);

			List<ArrayList<Double>> inithideW = bp.gethideW();

			System.out.println("the initial hide weight----------------");
			print(inithideW);

			// train until error less than expected error
			while (true) {
				i++;
				bp.train();
				if (i == 1) {
					firstErr = bp.error;
				}
				if (bp.error < error) {
					finalErr = bp.error;
					numBatch = i;
					break;
				}
				if (i >= 100000) {
					numBatch = 100000;
					finalErr = bp.error;
					break;
				}
			}

			System.out.println("first-batch error-----------------------");
			System.out.println(firstErr);
			System.out.println("final input node weight-----------------");
			print(bp.getinW());
			System.out.println("final hide node weight------------------");
			print(bp.gethideW());
			System.out.println("final error-----------------------------");
			System.out.println(finalErr);
			System.out.println("total number of batches-----------------");
			System.out.println(numBatch);
			
			
		}
	}

	public static void print(List<ArrayList<Double>> list) {
		for (int i = 0; i < list.size(); i++) {
			for (int j = 0; j < list.get(i).size(); j++) {
				System.out.print(list.get(i).get(j) + "  ");
			}
			System.out.println();
		}
	}
}
