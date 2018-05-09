import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.StringTokenizer;

/*
 * Data Mining 
 * Assignment4
 * 04/20/2017
 * Naive Bayes and KNN
 */

public class DataMining {

    static File set;
    static File[] dataset;

    static int spamTotal = 0; //total number of words in spam emails
    static int regTotal = 0; //total number of words in regular emails
    
    static double trainingdata[][];
    static double testingingdata[][];

    static Map<String, Integer> spam = new HashMap<String, Integer>();
    static Map<String, Integer> reg = new HashMap<String, Integer>();
    static Map<String, Integer> test = new HashMap<String, Integer>();
    static Map<String, Integer> tempSpam = new HashMap<String, Integer>();
    static Map<String, Integer> tempReg = new HashMap<String, Integer>();

    //build classifiers based on training data set
    public static void buildClassifier() throws IOException {
        // take in train files
        set = new File(System.getProperty("user.dir") + "/train");
        dataset = set.listFiles();

        String type;

        for (File f : dataset) { // loop through folder

            if (f.getName().startsWith("sp")) {
                type = "spam";
            } else {
                type = "reg";
            }
            train(f, type); // train the train files
        }

        // print to external text file for debugging purposes
        // print(spam, "spam");
        // print(reg, "not_spam");
    }

    //train the data set passed in (create hashmaps) 
    public static void train(File name, String type) throws FileNotFoundException {

        Map<String, Integer> map = new HashMap<String, Integer>();

        if (type.equals("spam")) {
            map = spam;
        } else if (type.equals("reg")) {
            map = reg;
        } else if (type.equals("test")) {
            map = test;
        }

        Scanner scan = new Scanner(name);
        while (scan.hasNext()) {

            String word = scan.nextLine();
            StringTokenizer st = new StringTokenizer(word);

            while (st.hasMoreTokens()) {
                String str = st.nextToken();

                if (type.equals("spam")) { 
                    spamTotal++;
                } else if (type.equals("reg")) {
                    regTotal++;
                }
                
                if (map.containsKey(str)) {
                    // increment word frequency if the word already exist
                    // in Hashmap
                    map.put(str, map.get(str) + 1);
                } else {

                    map.put(str, 1); // initialize each word with value of 1
                }
            }
        }

        scan.close();
    }

    //print hashmap to external files for debugging purposes
    public static void print(Map<String, Integer> type, String name) throws IOException {

        PrintWriter s = new PrintWriter(new FileWriter(name + ".txt"));

        for (Map.Entry<String, Integer> entry : type.entrySet()) {
            s.write(entry.getKey() + ": " + entry.getValue() + "\n");
        }
        s.close();
    }

    //calculate probability
    public static double calcProb(int freq, int total) {
        double constant = (spam.size() + reg.size());
        double prob = (((double) freq) + 1) / (total + constant);
        return prob;
    }

    //check to see if word exist in spam emails
    public static int existInSpam(String testword, boolean filter) {

        if (filter) { //filtered - loop through hashmap containing words > 50 counts
            for (Map.Entry<String, Integer> entry : tempSpam.entrySet()) {

                if (testword.equals(entry.getKey())) { 
                    return entry.getValue(); //return the frequency of that word
                }
            }
            return 0;
        } else { //unfiltered - loop through hashmap containing all spam words
            for (Map.Entry<String, Integer> entry : spam.entrySet()) {

                if (testword.equals(entry.getKey())) {
                    return entry.getValue(); //return the frequency of that word
                }
            }
            return 0;
        }

    }

    //check to see if word exist in regular emails
    public static int existInReg(String testword, boolean filter) {

        if (filter) { //filtered - loop through hashmap containing words > 50 counts
            for (Map.Entry<String, Integer> entry : tempReg.entrySet()) {

                if (testword.equals(entry.getKey())) {
                    return entry.getValue(); //return the frequency of that word
                }

            }
            return 0;
        } else { //unfiltered - loop through hashmap containing all regular words
            for (Map.Entry<String, Integer> entry : reg.entrySet()) {

                if (testword.equals(entry.getKey())) {
                    return entry.getValue(); //return the frequency of that word
                }

            }
            return 0;
        }

    }

    //calculate naive bayes
    public static int naiveBayes(boolean filter) throws IOException {

        String type;

        // take in test files
        set = new File(System.getProperty("user.dir") + "/test");
        dataset = set.listFiles();

        int correct = 0;

        for (File f : dataset) { // loop through folder

            double sumTest = 0.0;

            // System.out.println(f.getName());

            if (f.getName().startsWith("sp")) { //determine if file is spam or not
                type = "spam";
            } else {
                type = "reg";
            }

            // System.out.println("this is a " + type + " file");

            train(f, "test"); // train the test files

            // scan each each word in hashmap
            for (Map.Entry<String, Integer> entry : test.entrySet()) {

                // calculate the probability of the word appear in spam
                double spamProb = calcProb(existInSpam(entry.getKey(), filter), spamTotal);
                // calculate the probability of the word appear in regular
                double regProb = calcProb(existInReg(entry.getKey(), filter), regTotal);
                // take the log of the quotient
                sumTest += Math.log10(spamProb / regProb);
            }

            //class probabilities
            double PS = (double) spamTotal / (regTotal + spamTotal);
            double PH = (double) regTotal / (regTotal + spamTotal);

            //add class probabilities 
            sumTest += Math.log10(PS / PH);

            String t = type;

            if (sumTest > 0) { //spam
                t = "spam";
            }

            if (sumTest < 0) { //not spam
                t = "reg";
            }

            if (t.equals(type)) { //compare to original file
                correct++;
            }
            // System.out.println("classified: " + t);
            // System.out.println();

            //point to new hashmap because we don't want to keep add words 
            //the same hashmap
            test = new HashMap<String, Integer>();
            
        } //end for
        return correct;
    }

    //reset every static variables
    public static void reset() {
        set = null;
        spamTotal = 0;
        regTotal = 0;
        dataset = null;
        spam = new HashMap<String, Integer>();
        reg = new HashMap<String, Integer>();
        test = new HashMap<String, Integer>();
        tempSpam = new HashMap<String, Integer>();
        tempReg = new HashMap<String, Integer>();

    }

    //train the dataset passed in with all special characters removed
    public static void preProcess(File name, String type) throws IOException {
        Map<String, Integer> map = new HashMap<String, Integer>();

        if (type.equals("spam")) {
            map = spam;
        } else if (type.equals("reg")) {
            map = reg;
        } else if (type.equals("test")) {
            map = test;
        }

        Scanner scan = new Scanner(name);
        while (scan.hasNext()) {

            String word = scan.nextLine();
            StringTokenizer st = new StringTokenizer(word);

            while (st.hasMoreTokens()) {
                String str = st.nextToken();
                switch (str) { //removing all special characters
                case "-":
                    continue;
                case ".":
                    continue;
                case "?":
                    continue;
                case ">":
                    continue;
                case ";":
                    continue;
                case "+":
                    continue;
                case ",":
                    continue;
                case "&":
                    continue;
                case "@":
                    continue;
                case "#":
                    continue;
                case "$":
                    continue;
                case "*":
                    continue;
                case "(":
                    continue;
                case ")":
                    continue;
                case "_":
                    continue;
                case "=":
                    continue;
                case "/":
                    continue;
                case "<":
                    continue;
                case "|":
                    continue;
                case "^":
                    continue;
                case "%":
                    continue;
                case "!":
                    continue;
                case "~":
                    continue;
                case "`":
                    continue;
                case "\"":
                    continue;
                case "]":
                    continue;
                case "[":
                    continue;
                case "}":
                    continue;
                case "{":
                    continue;
                case ":":
                    continue;
                case "\\":
                    continue;
                case "'":
                    continue;
                default:

                    if (type.equals("spam")) {
                        spamTotal++;
                    } else if (type.equals("reg")) {
                        regTotal++;
                    }

                    if (map.containsKey(str)) {
                        // increment word frequency if the word already exist
                        // in Hashmap
                        map.put(str, map.get(str) + 1);
                    } else {
                        map.put(str, 1); // initialize each word with value of 1
                    }
                }
            }
        }

        scan.close();
    }

    //add all words that appear more than 50 times to a new hashmap
    public static boolean greaterThan50Freq(Map<String, Integer> type, String check) {
        boolean filter = false;
        for (Map.Entry<String, Integer> entry : type.entrySet()) {

            if (entry.getValue() > 50) {
                filter = true;
                if (check.equals("spam")) {
                    tempSpam.put(entry.getKey(), entry.getValue());
                } else {
                    tempReg.put(entry.getKey(), entry.getValue());
                }
            }
        }
        return filter;
    }

    //calculate knn
    public static double KNN(double[][] traindata, double[][] testdata, double[] trainclasses, double[] testclass,
            int trainheight, int trainwidth, int testheight, int testwidth, int k) {
        // Traindata and testdata are two by two arrays with rows being
        // independent records and columns being attributes,
        // the order matters for the columns and must be the same for both test
        // data and train data
        // trainclass and testclass are the class of the the records from test
        // and training data, they match up one to one with the rows in training
        // and test data
        // trainheight and test height is number of training and test records
        // respectively
        // trainwidth and testwidth are the number of unique attributes the
        // training and test records have. K is a constant decided by the user.
        double[] newclass = new double[(int) testheight];// the array we will
                                                         // put our new classes
                                                         // into
        double accuracy = 0; // what we will return
        double accurate = 0; // number accurately predicted classes
        double predictions = 0; // number of precicted classes

        // classifying
        for (int i = 0; i < testheight; i++) { // for each of the test point
            double[] distance = new double[trainheight]; // makes a count for
                                                         // probably of c1
            for (int j = 0; j < trainheight; j++) {// for each training data
                double distanceholder = 0; // number of times its of class 0
                for (int z = 0; z < testwidth; z++) {// for each attribute
                    if (j < trainwidth) { // checks the if were entering words
                                          // the training data doesn't have
                        distanceholder = distanceholder + Math.pow((testdata[i][z] - traindata[j][z]), 2);
                    }
                }
            }
        }
        // calculates accuracy
        System.out.println("Number of accurate" + accurate);
        System.out.println("Number of predictions" + predictions);
        accuracy = accurate / predictions; // calculate accuracy
        return accuracy;
    }

    public static void main(String[] args) throws IOException {

        boolean filter = false;

        System.out.println("-----------Naive Bayes-----------\n");
        System.out.println("Building classifiers...");

        buildClassifier();

        System.out.println("Classifiers built");
        System.out.println("Classifying each test record...");

        double accuracy = (double) naiveBayes(filter) / dataset.length;

        System.out.println("Accuracy on the test set is " + accuracy);

        reset(); //reset all static variables for better performance

        ///////////////////////////////////////////////////////////

        set = new File(System.getProperty("user.dir") + "/train");
        dataset = set.listFiles();
        String type;

        for (File f : dataset) { // loop through folder

            if (f.getName().startsWith("sp")) {
                type = "spam";
            } else {
                type = "reg";
            }
            preProcess(f, type); // train the train files
        }

        if (greaterThan50Freq(spam, "spam") && greaterThan50Freq(reg, "reg")) {
            filter = true;
        }

        accuracy = (double) naiveBayes(filter) / dataset.length;
        System.out.println("Accuracy after filtering is " + accuracy);
        reset();
        
    }

}
