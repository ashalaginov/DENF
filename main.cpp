/* 
 * File:   main.cpp
 * \brief: Implementation of the fuzzy rules construction module using neuro-fuzzy approach. 
 * \brief: Also Dynamic Expansion Neuro-Fuzzy (DENF) by adding a single term in the first fuzzy set, so 4 terms in the first one. 
 * \brief: Then re-training and check for accuracy.
 * \brief: Fuzzification into 2, 3 5 and 7 predefined terms in each of the fuzzy sets
 * \author Andrey Shalaginov <andrii.shalaginov@hig.no>
 * Created on September 25, 2014, 12:45 AM
 */

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdexcept>
#include <omp.h> //OpenMP
#include <limits.h>

//Include STL
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm> // std::find

#define nDataSamples  604 //Size of the input data set

//Fuzzy logic constants
#define nFuzzySets 6 // Equal to number of features on the input
#define nFuzzyTerms 3 //Can be 2, 3, 5 or 7
#define numerSelectedRules 10 //Number of rules to be selected out of all constructed
#define numerSelectedRulesDelta 10 //Number of rules to be selected out of all newly constructed

//ANN Terms
#define learningRateAlpha 0.1 //Learning rate in the Artificial Neural network
#define nEpochs 100 //Number of epochs to train ANN

#define maxThreads 8

using namespace std;

/**
 * Sigmoid activation function
 * @param x linear combiner value
 * @return activation function values
 */
double sigmoidFunction(double x) {
    return 1 / (1 + exp(-x));
}

/**
 * Fuzzification function using 3 or 5 linguistic terms
 * @param x value to be fuzzified
 * @param stDev standard deviation of the dataset
 * @param mean mean of the dataset
 * @param idSet ID of the linguistic term
 * @return membership degree of the value x
 */
double fuzzificationFunction(double x, double stDev, double mean, char idSet) {

    //TODO:: interval range!
    double tmp1;
    if (nFuzzyTerms == 2) {
        if (idSet == 0)
            tmp1 = pow((x - mean - stDev) / (stDev), 2);
        else if (idSet == 1)
            tmp1 = pow((x - mean + stDev) / (stDev), 2);
    } else if (nFuzzyTerms == 3) {
        if (idSet == 0)
            tmp1 = pow((x - mean - 2 * stDev) / (stDev), 2);
        else if (idSet == 1)
            tmp1 = pow((x - mean) / (stDev), 2);
        else if (idSet == 2)
            tmp1 = pow((x - mean + 2 * stDev) / (stDev), 2);
    } else if (nFuzzyTerms == 5) {
        if (idSet == 0)
            tmp1 = pow((x - mean - 3 * stDev) / (stDev), 2);
        else if (idSet == 1)
            tmp1 = pow((x - mean - 1 * stDev) / (stDev), 2);
        else if (idSet == 2)
            tmp1 = pow((x - mean) / (stDev), 2);
        else if (idSet == 3)
            tmp1 = pow((x - mean + 1 * stDev) / (stDev), 2);
        else if (idSet == 4)
            tmp1 = pow((x - mean + 3 * stDev) / (stDev), 2);
    } else if (nFuzzyTerms == 7) {
        if (idSet == 0)
            tmp1 = pow((x - mean - 3 * stDev) / (stDev), 2);
        else if (idSet == 1)
            tmp1 = pow((x - mean - 2 * stDev) / (stDev), 2);
        else if (idSet == 2)
            tmp1 = pow((x - mean - 1 * stDev) / (stDev), 2);
        else if (idSet == 3)
            tmp1 = pow((x - mean) / (stDev), 2);
        else if (idSet == 4)
            tmp1 = pow((x - mean + 1 * stDev) / (stDev), 2);
        else if (idSet == 5)
            tmp1 = pow((x - mean + 2 * stDev) / (stDev), 2);
        else if (idSet == 6)
            tmp1 = pow((x - mean + 3 * stDev) / (stDev), 2);
    }

    return 1 / exp(tmp1);
}

/**
 * For the adding new term, which has index 4
 */
double fuzzificationFunctionNew(double x, double stDev, double mean, char idSet) {
    double tmp1;
    if (idSet == 0)
        tmp1 = pow((x - mean - 1 * stDev) / (stDev), 2);
    else if (idSet == 1)
        tmp1 = pow((x - mean - 0.5 * stDev) / (stDev), 2);
    else if (idSet == 2)
        tmp1 = pow((x - mean) / (stDev), 2);
    else if (idSet == 3)
        tmp1 = pow((x - mean + 0.5 * stDev) / (stDev), 2);
    else if (idSet == 4)
        tmp1 = pow((x - mean + 1 * stDev) / (stDev), 2);
    return 1 / exp(tmp1);
}

/**
 * SupplementaryFunction for determining the linguistic term name based on its ID
 * @param id ID of the linguistic term
 * @return name of linguistic term
 */
string idToLingustic(short int id) {
    string strTmp = "";
    if (nFuzzyTerms == 2) {
        if (id == 0)
            strTmp = "   LOW";
        else if (id == 1)
            strTmp = "HIGH";
    } else
        if (nFuzzyTerms == 3) {
        if (id == 0)
            strTmp = "   LOW";
        else if (id == 1)
            strTmp = "MEDIUM";
        else if (id == 2)
            strTmp = "  HIGH";
    } else if (nFuzzyTerms == 5) {
        if (id == 0)
            strTmp = " VERY LOW";
        else if (id == 1)
            strTmp = "      LOW";
        else if (id == 2)
            strTmp = "   MEDIUM";
        else if (id == 3)
            strTmp = "     HIGH";
        else if (id == 4)
            strTmp = "VERY HIGH";
    } else if (nFuzzyTerms == 7) {
        if (id == 0)
            strTmp = " EXTREME LOW";
        else if (id == 1)
            strTmp = " VERY LOW";
        else if (id == 2)
            strTmp = "      LOW";
        else if (id == 3)
            strTmp = "   MEDIUM";
        else if (id == 4)
            strTmp = "     HIGH";
        else if (id == 5)
            strTmp = "VERY HIGH";
        else if (id == 6)
            strTmp = "EXTREME HIGH";
    }
    return strTmp;
}


/**
 * For the adding new term, which has index 4
 */
string idToLingusticNew(short int id) {
    string strTmp = "";
    if (id == 0)
        strTmp = " VERY LOW";
    else if (id == 1)
        strTmp = "      LOW";
    else if (id == 2)
        strTmp = "   MEDIUM";
    else if (id == 3)
        strTmp = "     HIGH";
    else if (id == 4)
        strTmp = "VERY HIGH";
    return strTmp;
}

/**
 * Main function
 * @param argc
 * @param argv
 * @return 
 */
int main(int argc, char** argv) {
    vector< vector<double> > input, inputTest; //train and test datasets patterns
    vector<double> tmp; //temporary variable
    double tmp1, tmp2; //temporary variables
    vector<double> classID, classIDTest; //train and test classes
    vector<double> means, means1, means2; //means for all dataset and for both classes
    vector<double> stDev, stDev1, stDev2; //StDev for all dataset and for both classes
    long long int numberRules = (long long int) pow(nFuzzyTerms, nFuzzySets); //total amount of extracted fuzzy rules
    vector<double> weights(numberRules); //weights of each rule
    short int cl; //temporary variable
    vector<double> rules(numberRules); //extracted fuzzy rules
    double outputOldBuffer[nEpochs][nDataSamples];
    int SRules = numerSelectedRules; //Tmp for plot building
    //Batch execution, so accepts the first argument - number of rules to select.
    if (argc > 1) {
        SRules = atoi(argv[1]);
#undef numerSelectedRules
#define numerSelectedRules SRules
    }


    FILE *pFileTrain, *pFileTest; //pointer to train and test files

    puts("Reading files...");
    //Read files with training and testing datasets
    if ((pFileTrain = fopen("data/6features.txt", "rt")) == NULL)
        puts("Error while opening input train file!");
    if ((pFileTest = fopen("data/6features_test.txt", "rt")) == NULL)
        puts("Error while opening input test file!");

    //Parsing train file content into data structure
    while (!feof(pFileTrain)) {
        tmp.clear();
        for (int i = 0; i < nFuzzySets; i++) {
            fscanf(pFileTrain, "%lf ", &tmp1);
            tmp.push_back((double) tmp1);
        }
        input.push_back(tmp);
        fscanf(pFileTrain, "%hd ", &cl);
        classID.push_back(cl);
    }

    //Parsing test file content into data structure
    while (!feof(pFileTest)) {
        tmp.clear();
        for (int i = 0; i < nFuzzySets; i++) {
            fscanf(pFileTest, "%lf ", &tmp1);
            tmp.push_back((double) tmp1);

        }
        inputTest.push_back(tmp);
        fscanf(pFileTest, "%hd ", &cl);
        classIDTest.push_back(cl);
    }

    //Frees & Closes
    fclose(pFileTrain);
    fclose(pFileTest);
    puts("Statistics calculation...");
    //Calculate mean and standard deviation for both classes separately
    //Mean
    int numEntriesTmp1, numEntriesTmp2;
    for (int j = 0; j < nFuzzySets; j++) {

        tmp1 = 0;
        tmp2 = 0;
        numEntriesTmp1 = 0;
        numEntriesTmp2 = 0;
        for (int i = 0; i < nDataSamples; i++) {
            if (classID[i] == 0) {
                tmp1 += input[i][j];
                numEntriesTmp1++;
            } else if (classID[i] == 1) {
                tmp2 += input[i][j];
                numEntriesTmp2++;
            }
        }
        tmp1 = tmp1 / numEntriesTmp1;
        means1.push_back(tmp1);
        tmp2 = tmp2 / numEntriesTmp2;
        means2.push_back(tmp2);
    }

    //StDev
    for (int j = 0; j < nFuzzySets; j++) {
        tmp1 = 0;
        tmp2 = 0;
        numEntriesTmp1 = 0;
        numEntriesTmp2 = 0;
        for (int i = 0; i < nDataSamples; i++) {
            if (classID[i] == 0) {
                tmp1 += (input[i][j] - means1[j])*(input[i][j] - means1[j]);
                numEntriesTmp1++;

            } else if (classID[i] == 1) {
                tmp2 += (input[i][j] - means2[j])*(input[i][j] - means2[j]);
                numEntriesTmp2++;

            }
        }
        tmp1 = sqrt(tmp1 / numEntriesTmp1);
        stDev1.push_back(tmp1);
        tmp2 = sqrt(tmp2 / numEntriesTmp2);
        stDev2.push_back(tmp2);
    }

    //Calculate overall mean and standard deviation
    //Mean
    for (int j = 0; j < nFuzzySets; j++) {
        tmp1 = 0;
#pragma omp parallel for reduction(+:tmp1)  num_threads(maxThreads)
        for (int i = 0; i < nDataSamples; i++) {
            tmp1 += input[i][j];
        }
        tmp1 = tmp1 / nDataSamples;
        means.push_back(tmp1);
    }

    //StDev
    for (int j = 0; j < nFuzzySets; j++) {
        tmp1 = 0;
#pragma omp parallel for reduction(+:tmp1)  num_threads(maxThreads)
        for (int i = 0; i < nDataSamples; i++) {
            tmp1 += (input[i][j] - means[j])*(input[i][j] - means[j]);

        }
        tmp1 = sqrt(tmp1 / (nDataSamples));
        stDev.push_back(tmp1);
    }

    //Weights Initialization
#pragma omp parallel for   num_threads(maxThreads)
    for (long long int j = 0; j < numberRules; j++)
        //weights.push_back(1 / (numberRules));
        weights[j] = 0.5;


    
    //----------------------MLP Learning----------------------------------------
    puts("MLP Learning...");
    clock_t begin = clock();
    for (int i = 0; i < nEpochs; i++) {
        //For each input patern
        for (int m = 0; m < nDataSamples; m++) {
            //Assigning degree of membership 
            double output = 0;
#pragma omp parallel for reduction(+:output)  num_threads(maxThreads)
            for (long long int j = 0; j < numberRules; j++) {
                double tmp1 = 1;
                //Calculation of the membership function, which is multiplication of the membership function of each term in the rule
                tmp1 = fuzzificationFunction(input[m][0], means[0], stDev[0], (int) j / (int) pow(nFuzzyTerms, nFuzzySets - 1));
                for (int l = 1; l < nFuzzySets; l++)
                    tmp1 *= fuzzificationFunction(input[m][l], means[l], stDev[l], (int) j / (int) pow(nFuzzyTerms, nFuzzySets - l - 1) % (int) pow(nFuzzyTerms, 1));
                rules[j] = tmp1;
                output += weights[j] * rules[j];
            }

            //Adjusting weights using the Delta Learning Rule
#pragma omp parallel for num_threads(maxThreads)
            for (long long int k = 0; k < numberRules; k++)
                weights[k] += learningRateAlpha * (classID[m] - sigmoidFunction(output)) * sigmoidFunction(output)*(1 - sigmoidFunction(output)) * rules[k];
            // printf("%f \n", weights[0]);
            outputOldBuffer[i][m] = output;
        }
    }

    clock_t end = clock();
    printf("\nElasped time is %.8lf seconds.", double(end - begin) / (CLOCKS_PER_SEC * maxThreads));


    //------------------------RULES SELECTION-----------------------------------
    puts("Rules selection started...");
    //Sorting constructed rules according to fuzzy-neuro weight value
    vector<short int> rulesAtomId;
    map<double, vector<short int> > rulesMap;
    for (long long int k = 0; k < numberRules; k++) {
        rulesAtomId.clear();
        rulesAtomId.push_back((int) k / (int) pow(nFuzzyTerms, nFuzzySets - 1)); //initial one
        //pushing ID of the term for each fuzzy set into the vector
        for (int l = 1; l < nFuzzySets; l++)
            rulesAtomId.push_back((int) k / (int) pow(nFuzzyTerms, nFuzzySets - l - 1) % (int) pow(nFuzzyTerms, 1));

        //push into format <weight, <term1, term2....>> into map, so it is sorted according to the weights value (significance)
        rulesMap.insert(std::pair<double, vector<short int> >(weights[k] + 1e-5 * k, rulesAtomId)); //obfuscation is needed
    }


    //Assigning the Class ID to extracted rules based on MIN-MAX principle
    //Set vector with corresponding <class> to each <weight, <term1, term2....>>
    std::map<double, vector<short int> >::reverse_iterator it;
    vector< vector<short int> >extractedRules; //Set with extracted rules denoted by ID
    vector<short int>extractedRulesClasses; //Set of corresponding classes for each rule
    vector<double>extractedRulesWeights; //ANN weight for each corresponding rule
    vector<short int> ruleTmp;
    for (it = rulesMap.rbegin(); it != rulesMap.rend(); it++) {
        ruleTmp.clear();
        double degC1 = 1, degC2 = 1;
        //Calculation of membership degree for each class
        for (int l = 0; l < nFuzzySets; l++) {
            degC1 *= fuzzificationFunction(means1[l], stDev[l], means[l], it->second[l]);
            degC2 *= fuzzificationFunction(means2[l], stDev[l], means[l], it->second[l]);
            ruleTmp.push_back(it->second[l]);
        }
        //Defining class
        if (degC1 > degC2) 
            extractedRulesClasses.push_back(0);
         else 
            extractedRulesClasses.push_back(1);
        extractedRules.push_back(ruleTmp);
        extractedRulesWeights.push_back(it->first);
    }

    //Erase irrelevant rules
    if (numerSelectedRules < numberRules) {
        long int tmpSelectRules = floor(numerSelectedRules / 2);
        extractedRules.erase(extractedRules.begin() + tmpSelectRules, extractedRules.end() - tmpSelectRules);
        extractedRulesClasses.erase(extractedRulesClasses.begin() + tmpSelectRules, extractedRulesClasses.end() - tmpSelectRules);
        extractedRulesWeights.erase(extractedRulesWeights.begin() + tmpSelectRules, extractedRulesWeights.end() - tmpSelectRules);
    }

    //Print constructed rules
    printf("rule weight |METRICpermissions|METRICstatic| METRICsdk  |  METRICresources  |METRICdynamic | Class  (m.deg Cl1  m.deg Cl2)\n");
    for (long int p = 0; p < extractedRules.size(); p++) {
        printf("w: %.6f | ", extractedRulesWeights[p]);
        double degC1 = 1, degC2 = 1;
        for (int l = 0; l < nFuzzySets; l++) {
            if (l > 0)
                printf(" and ");
            printf("%s  ", idToLingustic(extractedRules[p][l]).c_str());
            degC1 *= fuzzificationFunction(means1[l], stDev[l], means[l], extractedRules[p][l]);
            degC2 *= fuzzificationFunction(means2[l], stDev[l], means[l], extractedRules[p][l]);
        }
        //printing class of the rule
        if (degC1 > degC2)
            printf("=> Class0");
        else
            printf("=> Class1");

        printf(" (%f     ", degC1);
        printf(" %f    )\n", degC2);
    }

    printf("\nAmount of constructed rules: %d", (int) pow(nFuzzyTerms, nFuzzySets));
    printf("\nAmount of selected rules: %ld", extractedRulesClasses.size());

    printf("\nElasped time is %.lf seconds.", double(clock() - end) / (CLOCKS_PER_SEC * maxThreads));
    end = clock();

    vector< vector<short int> >extractedRulesNew;
    vector<short int>extractedRulesClassesNew;
    vector<double>extractedRulesWeightsNew;
    vector<short int> ruleTmpNew;




    //----------------ADD a single term-----------------------------------------
    puts("\nMLP Re-training after adding term VERY HIGH in fuzzy set 0...");

    //Fuzzy Set=0, Term=4 (VERY HIGH) to existing 3 terms fuzzy sets
    
    //Do not retrain existing number of rules, just train new set of rules
    long long int numberRulesNew = (long long int) pow(nFuzzyTerms, nFuzzySets - 1); //total amount of extracted fuzzy rules
    vector<double> weightsNew(numberRulesNew); //weights of each rule
    vector<double> rulesNew(numberRulesNew); //extracted fuzzy rules

    //Weights Initialization
#pragma omp parallel for   num_threads(maxThreads)
    for (long long int j = 0; j < numberRulesNew; j++)
        weightsNew[j] = 0.5;

    //Retrain network only for new rules that includes new term
    for (int i = 0; i < nEpochs; i++) {
        //For each input pattern
        for (int m = 0; m < nDataSamples; m++) {
            //Assigning degree of membership 
            //Use the initial output as reference and add values from the new rules
            double output = outputOldBuffer[i][m];
            //NEW rules construction and weight calculation 
#pragma omp parallel for reduction(+:output)  num_threads(maxThreads)
            for (long long int j = 0; j < numberRulesNew; j++) {
                double tmp1 = 1;
                //Membership calculation suing a New fuzzification function for the first fuzzy set
                tmp1 = fuzzificationFunctionNew(input[m][0], means[0], stDev[0], 4);
                for (int l = 1; l < nFuzzySets; l++)
                    tmp1 *= fuzzificationFunction(input[m][l], means[l], stDev[l], (int) j / (int) pow(nFuzzyTerms, nFuzzySets - l - 1) % (int) pow(nFuzzyTerms, 1));
                rulesNew[j] = tmp1;
                output += weightsNew[j] * rulesNew[j];
            }
     

            //Adjusting weights of new added rules
            //since OLD model to be kept old, so no adjustment in the old model
#pragma omp parallel for num_threads(maxThreads)
            for (long long int k = 0; k < numberRulesNew; k++)
                weightsNew[k] += learningRateAlpha * (classID[m] - sigmoidFunction(output)) * sigmoidFunction(output)*(1 - sigmoidFunction(output)) * rulesNew[k];

        }
    }



    printf("\nElasped time is %.8lf seconds.", double(clock() - end) / (CLOCKS_PER_SEC * maxThreads));
    end = clock();

    puts("\nNew Rules selection started...");
    //Sorting constructed rules according to fuzzy-neuro weight value
    vector<short int> rulesAtomIdNew;
    map<double, vector<short int> > rulesMapNew;
    for (long long int k = 0; k < numberRulesNew; k++) {
        rulesAtomIdNew.clear();
        rulesAtomIdNew.push_back(4); //initial one
        //pushing ID of the term for each fuzzy set into the vector
        for (int l = 1; l < nFuzzySets; l++) {
            rulesAtomIdNew.push_back((int) k / (int) pow(nFuzzyTerms, nFuzzySets - l - 1) % (int) pow(nFuzzyTerms, 1));
        }
        //push into format <weight, <term1, term2....>>
        rulesMapNew.insert(std::pair<double, vector<short int> >(weightsNew[k] + 1e-5 * k, rulesAtomIdNew)); //obfuscation is needed
    }



    //NEW
    //Assigning the Class ID to extracted rules based on MIN-MAX principle
    //Set vector with corresponding <class> to each <weight, <term1, term2....>>
    std::map<double, vector<short int> >::reverse_iterator itNew;

    for (itNew = rulesMapNew.rbegin(); itNew != rulesMapNew.rend(); itNew++) {
        ruleTmpNew.clear();
        double degC1New = 1, degC2New = 1;
        //Calculation of membership degree for each class
        //First one
        degC1New *= fuzzificationFunctionNew(means1[0], stDev[0], means[0], 4);
        degC2New *= fuzzificationFunctionNew(means2[0], stDev[0], means[0], 4);
        ruleTmpNew.push_back(itNew->second[0]);

        //Other
        for (int l = 1; l < nFuzzySets; l++) {
            degC1New *= fuzzificationFunction(means1[l], stDev[l], means[l], itNew->second[l]);
            degC2New *= fuzzificationFunction(means2[l], stDev[l], means[l], itNew->second[l]);
            ruleTmpNew.push_back(itNew->second[l]);
        }
        //Defining class
        if (degC1New > degC2New) {
            extractedRulesClassesNew.push_back(0);
        } else {

            extractedRulesClassesNew.push_back(1);
        }
        extractedRulesNew.push_back(ruleTmpNew);
        extractedRulesWeightsNew.push_back(itNew->first);
    }


    //NEW
    //Erase irrelevant rules from the New rules set
    if (numerSelectedRulesDelta < numberRulesNew) {
        long int tmpSelectRulesNew = floor(numerSelectedRulesDelta / 2);
        extractedRulesNew.erase(extractedRulesNew.begin() + tmpSelectRulesNew, extractedRulesNew.end() - tmpSelectRulesNew);
        extractedRulesClassesNew.erase(extractedRulesClassesNew.begin() + tmpSelectRulesNew, extractedRulesClassesNew.end() - tmpSelectRulesNew);
        extractedRulesWeightsNew.erase(extractedRulesWeightsNew.begin() + tmpSelectRulesNew, extractedRulesWeightsNew.end() - tmpSelectRulesNew);
    }


    //NEW   
    //Print constructed rules
    printf("rule weight |METRICpermissions|METRICstatic| METRICsdk  |  METRICresources  |METRICdynamic | Class  (m.deg Cl1  m.deg Cl2)\n");
    //TODO: change on the names of variables
    for (long int p = 0; p < extractedRulesNew.size(); p++) {
        printf("w: %.6f | ", extractedRulesWeightsNew[p]);
        double degC1New = 1, degC2New = 1;

        printf("%s  ", idToLingusticNew(extractedRulesNew[p][0]).c_str());
        degC1New *= fuzzificationFunctionNew(means1[0], stDev[0], means[0], 4);
        degC2New *= fuzzificationFunctionNew(means2[0], stDev[0], means[0], 4);
        for (int l = 1; l < nFuzzySets; l++) {
            printf(" and ");
            printf("%s  ", idToLingustic(extractedRulesNew[p][l]).c_str());
            degC1New *= fuzzificationFunction(means1[l], stDev[l], means[l], extractedRulesNew[p][l]);
            degC2New *= fuzzificationFunction(means2[l], stDev[l], means[l], extractedRulesNew[p][l]);
        }
        //printing class of the rule
        if (degC1New > degC2New)
            printf("=> Class0");
        else
            printf("=> Class1");

        printf(" (%f     ", degC1New);
        printf(" %f    )\n", degC2New);
    }

    printf("\nAmount of constructed rules: %d", (int) pow(nFuzzyTerms, nFuzzySets - 1));
    printf("\nAmount of selected rules: %ld", extractedRulesClassesNew.size());

    printf("\nElasped time is %.8lf seconds.", double(clock() - end) / (CLOCKS_PER_SEC * maxThreads));
    end = clock();



    puts("\nCross-validation");
    //--------------------Classification validation-----------------------------
    double classificationAccuracy = 0, maxFuz;
    short int actualClass, resClass0 = 0, resClass1 = 0, factClass0 = 0, factClass1 = 0;
    long int maxFuzId = 0;

    for (long int i = 0; i < inputTest.size(); i++) {
        //vector<map<double, short int > >linguistic_membership_var;
        // map<double, short int > lingusticAttributeSorted;
        ruleTmp.clear();
        //Checking the fuzzy rule for each input data sample
        maxFuz = -1;
        for (int m = 0; m < nFuzzyTerms + 2; m++) {
            //Check for the highest value
            if (maxFuz < fuzzificationFunctionNew(inputTest[i][0], means[0], stDev[0], m)) {
                maxFuzId = m;
                maxFuz = fuzzificationFunctionNew(inputTest[i][0], means[0], stDev[0], m);
            }

        }
        ruleTmp.push_back(maxFuzId);

        for (int l = 0; l < nFuzzySets; l++) {
            maxFuz = -1;
            for (int m = 0; m < nFuzzyTerms; m++) {

                //Check for the highest value
                if (maxFuz < fuzzificationFunction(inputTest[i][l], means[l], stDev[l], m)) {
                    maxFuzId = m;
                    maxFuz = fuzzificationFunction(inputTest[i][l], means[l], stDev[l], m);
                }
            }
            ruleTmp.push_back(maxFuzId);
        }

        //Searching the input pattern rule among previously constructed
        //1. USE fuzzification to define the closest class membership by distance, then derive the rule
        //2. calculate the shortest distance to the rule using the id of fuzzy terms!
        //3. check in the initial model
        double rulesMinDistance = DBL_MAX, //Ideally the distance between the input data pattern and rules should be 0
                tmpDistance = 0;
        long int rulesMinDistanceId =LONG_MIN;
        for (long int p = 0; p < extractedRules.size(); p++) {
            tmpDistance = 0;
            for (int l = 0; l < nFuzzySets; l++)
                tmpDistance += pow(ruleTmp[l] - extractedRules[p][l], 2);
            tmpDistance = sqrt(tmpDistance);
            if (tmpDistance < rulesMinDistance) {
                rulesMinDistance = tmpDistance;
                rulesMinDistanceId = p;
            }
        }

        //4. check in the delta (new) model
        int secondModelFlag = 0;
        for (long int p = 0; p < extractedRulesNew.size(); p++) {
            tmpDistance = 0;
            for (int l = 0; l < nFuzzySets; l++)
                tmpDistance += pow(ruleTmp[l] - extractedRulesNew[p][l], 2);
            tmpDistance = sqrt(tmpDistance);
            if (tmpDistance < rulesMinDistance) {
                rulesMinDistance = tmpDistance;
                rulesMinDistanceId = p;
               // secondModelFlag++;
            }
        }

        //Decide on the class
        if (secondModelFlag >= 1)
            actualClass = extractedRulesClassesNew[rulesMinDistanceId];
        else
            actualClass = extractedRulesClasses[rulesMinDistanceId];

        if (classIDTest[i] == actualClass) {
            classificationAccuracy++;
            if (classIDTest[i] == 1)
                resClass0++;
            else if (classIDTest[i] == 0)
                resClass1++;
        }
        
        if (classIDTest[i] == 1)
            factClass0++;
        else if (classIDTest[i] == 0)
            factClass1++;
    }

    printf("\nAccuracy: %.2f %% Total %d Class %d act %d Class1 %d act %d \n", classificationAccuracy / inputTest.size() * 100, nDataSamples, resClass0, factClass0, resClass1, factClass1);
    /*
        FILE *fp;
        fp = fopen("accuracy_5_1000epochs.txt", "a");
        fprintf(fp, "%d, %f\n", SRules, classificationAccuracy / inputTest.size() * 100);
        fclose(fp);
     */
    return 0;
}
