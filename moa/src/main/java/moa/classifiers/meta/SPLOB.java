/*
 *    SPLOB.java
 * 
 *    @author 
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */
package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.core.driftdetection.PageHinkleyDM;
import moa.classifiers.meta.AdaptiveRandomForest.ARFBaseLearner;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.AbstractMOAObject;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

import java.util.ArrayList;
import java.util.Random;


/**
 * Spaced Learning with Online Bagging
 *
 * <p>
 * Oversampling until the error decreases and undersampling when it becomes constant.
 * Moreover, use the Spaced Learning heuristic to better fix the learnt concept.
 * </p>
 *
 * <p>See details in:<br> </p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classifier to train</li>
 * <li>-S : The number of classifiers in the ensemble</li>
 * <li>-L : Starting lambda value to use in the Poisson distribution</li>
 * <li>-e : Seed value</li>
 * <li>-x : The change detector strategy to use for drifts</li>
 * <li>-s : Every how many samples to check for PH</li>
 * <li>-a : For how many samples to do oversampling.</li>
 * <li>-r : Reset model when drift occur, too.</li>
 * </ul>
 *
 * @author 
 * @version $Revision: 1 $
 */
public class SPLOB extends AbstractClassifier implements MultiClassClassifier,CapabilitiesHandler {

    @Override
    public String getPurposeString() {
        return "SPLOB doeas oversampling until the error decreases and then it does undersampling when the error becomes constant. Moreover, use the Spaced Learning heuristic to better fix the learnt concept";
    }
    
    private static final long serialVersionUID = 1L;
    
    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree -S");  
    
    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 'S',
            "The number of learners.", 1, 1, Integer.MAX_VALUE);  
       
    public FloatOption lambdaFixedOption = new FloatOption("lambdaFixed", 'L',
            "Fixed Lambda value to use to calculate # of samples.", 6.0, 1.0, Float.MAX_VALUE);
        
    public IntOption seedOption = new IntOption("seed", 'e',
            "Seed option.", 1, 1, Integer.MAX_VALUE); 
        
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
            "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-5");
    
    public IntOption patienceOption = new IntOption("patience", 's',
            "Every how many samples to check for PH.", 3000, 1, Integer.MAX_VALUE);
    
    public IntOption awakeningOption = new IntOption("awakening", 'a',
            "For how many samples to do oversampling.", 1000, 1, Integer.MAX_VALUE);
    
    public FlagOption resetModelOption = new FlagOption("resetModel", 'r',
            "Reset model when drift occur, too.");
    
    
    protected BaseLearner[] ensemble;           
    protected ArrayList<Integer> nInstances;
    protected ArrayList<ChangeDetector> driftDetectionMethod;                      
    protected double lambdaFixed;    
    protected ArrayList<Boolean> meanDifference;
    protected ArrayList<Boolean> pastMeanDifference;
    protected ArrayList<Integer> awakening;
    protected ArrayList<Integer> patience;
    protected ArrayList<PageHinkleyDM> pageHinkley;
            
    @Override
    public void resetLearningImpl() {
        // Reset attributes
    	this.ensemble = null;    	        
        this.classifierRandom = new Random(this.seedOption.getValue());                                                     
        this.lambdaFixed = this.lambdaFixedOption.getValue();          
        
        this.nInstances = new ArrayList<Integer>();
        this.driftDetectionMethod = new ArrayList<ChangeDetector>();
        this.meanDifference = new ArrayList<Boolean>();
        this.pastMeanDifference = new ArrayList<Boolean>();
        this.awakening = new ArrayList<Integer>();
        this.patience = new ArrayList<Integer>();
        this.pageHinkley = new ArrayList<PageHinkleyDM>();
        for (int i = 0; i < this.ensembleSizeOption.getValue(); i++) {        	   	                	
    		this.pageHinkley.add(new PageHinkleyDM());
    		this.meanDifference.add(i, true);
    		this.pastMeanDifference.add(i, null);
    		this.awakening.add(i, 0);
    		this.patience.add(i, this.patienceOption.getValue());
    		this.nInstances.add(i, 0);
    		this.driftDetectionMethod.add(((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy());
    	}        
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
    	if(this.ensemble == null) {
            initEnsemble(instance);
    	}    	    	    	                        
        Instance instanceFeatures = (Instance) instance.copy();        
        for (int i = 0 ; i < this.ensemble.length ; i++) {
        	this.nInstances.set(i, this.nInstances.get(i)+1);        	
	        double lambda = 0;               
	    	int treeSizeNodes = 0;
	    	int ensembleSize = 1;
	    	if (this.ensemble[i].classifier instanceof HoeffdingTree || this.ensemble[i].classifier instanceof HoeffdingAdaptiveTree) {    		
	    		treeSizeNodes = ((HoeffdingTree) this.ensemble[i].classifier).getTreeSizeNodes();
	    	}
	    	else if (this.ensemble[i].classifier instanceof AdaptiveRandomForest) {    		
	    		for (ARFBaseLearner element : ((AdaptiveRandomForest) this.ensemble[i].classifier).ensemble) {    			
	    			treeSizeNodes += element.classifier.getTreeSizeNodes();	    			    			 
				}    		
	    		ensembleSize = ((AdaptiveRandomForest) this.ensemble[i].classifier).ensemble.length;
	    	}
	    		    	
	    	this.pageHinkley.get(i).input((double)(treeSizeNodes / ensembleSize));
	    		    	
	    	if (this.nInstances.get(i) % this.patience.get(i) == 0) {    
	        	//return true if the means are different
	    		boolean change = this.pageHinkley.get(i).getChange();
	    		this.pastMeanDifference.set(i, this.meanDifference.get(i));
	        	this.meanDifference.set(i,change);
	        	this.pageHinkley.get(i).resetLearning();	        	
	        	//check if reactivate lambda with awakening option
	        	if (change == false && this.pastMeanDifference.get(i) == false) {
	        		this.awakening.set(i, this.awakeningOption.getValue());
	        		this.patience.set(i, this.patience.get(i)*2);
	        	} else if (change == true && this.pastMeanDifference.get(i) == false) {
	        		this.awakening.set(i, 0);
	        		this.patience.set(i, this.patienceOption.getValue());
	        	}
	        }
	    		    	
	    	//actual depth
	    	if (this.meanDifference.get(i) == true || this.awakening.get(i) > 0) {
	        	lambda = this.lambdaFixed;
	        	if (this.awakening.get(i) > 0) {
	        		this.awakening.set(i, this.awakening.get(i)-1);
	        	}
	        } else {            	       	            
	        	lambda = 0.1;            	
	        }	        	        
	
	    	int k = MiscUtils.poisson(lambda, this.classifierRandom); 	    	
	        if (k > 0) {      	        		        		        	
	        	this.ensemble[i].trainOnInstance(instanceFeatures,k);	        	
	        } 	    	
	        //drift detection in class distribution
	    	driftDetection(instance,i);
        }
    }  
    

    @Override
    public double[] getVotesForInstance(Instance instance) {   
    	if(this.ensemble == null) 
            initEnsemble(instance);
        
        DoubleVector combinedVote = new DoubleVector();
        Instance instanceFeatures = (Instance) instance.copy();
        for(int i = 0 ; i < this.ensemble.length ; ++i) {        	         	
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instanceFeatures));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();                
            	combinedVote.addValues(vote);                               
            }
        }                
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder arg0, int arg1) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
    	double treeSizeNodes = 0.0;
    	int ensembleSize = this.ensembleSizeOption.getValue();
    	
    	for (int i = 0 ; i < this.ensemble.length ; i++) {
    		if (this.ensemble[i].classifier instanceof HoeffdingTree || this.ensemble[i].classifier instanceof HoeffdingAdaptiveTree) {    		
        		treeSizeNodes += ((HoeffdingTree) this.ensemble[i].classifier).getTreeSizeNodes();
        	}
        	else if (this.ensemble[i].classifier instanceof AdaptiveRandomForest) {    		
        		for (ARFBaseLearner element : ((AdaptiveRandomForest) this.ensemble[i].classifier).ensemble) {    			
        			treeSizeNodes += element.classifier.getTreeSizeNodes();
        			    			 
    			}    		
        		int ensembleARF = ((AdaptiveRandomForest) this.ensemble[i].classifier).ensemble.length;
        		treeSizeNodes /= ensembleARF;
        	}
    	}    	
    	
    	return new Measurement[]{
                new Measurement("[avg] tree size (nodes)", treeSizeNodes/ensembleSize)};    	
    }

    
    protected void driftDetection(Instance instance, int i) {    	    	
    	boolean correctlyClassifies = this.ensemble[i].classifier.correctlyClassifies(instance);
    	// Update the DRIFT detection method
        this.driftDetectionMethod.get(i).input(correctlyClassifies ? 0 : 1);    	
        // Check if there was a change
        if (this.driftDetectionMethod.get(i).getChange()) {                    	       		
    		this.nInstances.set(i, 0);        		    		
        	this.driftDetectionMethod.get(i).resetLearning();        	        	
            this.meanDifference.set(i, true);
            this.pastMeanDifference.set(i, null);            
            this.pageHinkley.get(i).resetLearning();
            this.awakening.set(i, 0);
    		this.patience.set(i, this.patienceOption.getValue());
    		if (this.resetModelOption.isSet()) {    		
    			this.ensemble[i].reset();
    		}
        }	
    } 
    
    protected void initEnsemble(Instance instance) {        
    	// Init the ensemble.
    	int ensembleSize = this.ensembleSizeOption.getValue();     	
        this.ensemble = new BaseLearner[ensembleSize];        
        BasicClassificationPerformanceEvaluator classificationEvaluator = new BasicClassificationPerformanceEvaluator();                  
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        
        for(int i = 0 ; i < ensembleSize ; ++i) {        	
            this.ensemble[i] = new BaseLearner((Classifier) baseLearner.copy(),(BasicClassificationPerformanceEvaluator) classificationEvaluator.copy());
        }                        
    }   
    
    
    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == SPLOB.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    } 
    
    /**
     * Inner class that represents a single tree member of the forest. 
     * It contains some analysis information, such as the numberOfDriftsDetected, 
     */
    protected final class BaseLearner extends AbstractMOAObject {               
        public Classifier classifier;                
        public BasicClassificationPerformanceEvaluator evaluator;       
        

        private void init(Classifier instantiatedClassifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated) {                     
            this.classifier = instantiatedClassifier;
            this.evaluator = evaluatorInstantiated;            
        }

        public BaseLearner(Classifier instantiatedClassifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated) {
            init(instantiatedClassifier, evaluatorInstantiated);
        }

        public void reset() {            
          	this.classifier.resetLearning();                        
            this.evaluator.reset();
        }

        public void trainOnInstance(Instance instance, double weight) {        	
    		Instance weightedInstance = (Instance) instance.copy();
    		weightedInstance.setWeight(instance.weight() * weight);    		
            this.classifier.trainOnInstance(weightedInstance);       	                                 
        }

        public double[] getVotesForInstance(Instance instance) {
            DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
            return vote.getArrayRef();
        }        
        
        @Override
        public void getDescription(StringBuilder sb, int indent) {
        }
    }
}
