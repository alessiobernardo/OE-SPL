/*
 *    EvaluatePrequential.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.tasks;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;

import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.MultiClassClassifier;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.TimingUtils;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.evaluation.EWMAClassificationPerformanceEvaluator;
import moa.evaluation.FadingFactorClassificationPerformanceEvaluator;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.learners.Learner;
import moa.options.ClassOption;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import moa.streams.ExampleStream;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Utils;

/**
 * Task for evaluating a classifier on a stream by testing then training with each example in sequence.
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */
public class EvaluatePrequentialOvA extends ClassificationMainTask implements CapabilitiesHandler {

    @Override
    public String getPurposeString() {
        return "Evaluates a classifier on a stream by testing then training with each example in sequence.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption learnerOption = new ClassOption("learner", 'l',
            "Learner to train.", MultiClassClassifier.class, "moa.classifiers.bayes.NaiveBayes");

    public ClassOption streamOptionUp = new ClassOption("streamUp", 'U',
            "Stream 1 to learn from.", ExampleStream.class,
            "generators.RandomTreeGenerator");
    
    public ClassOption streamOptionDown = new ClassOption("streamDown", 'D',
            "Stream 2 to learn from.", ExampleStream.class,
            "generators.RandomTreeGenerator");
    
    public ClassOption streamOptionMiddle = new ClassOption("streamMiddle", 'M',
            "Stream 3 to learn from.", ExampleStream.class,
            "generators.RandomTreeGenerator");

    public ClassOption evaluatorOption = new ClassOption("evaluator", 'e',
            "Classification performance evaluation method.",
            LearningPerformanceEvaluator.class,
            "WindowClassificationPerformanceEvaluator");

    public IntOption instanceLimitOption = new IntOption("instanceLimit", 'i',
            "Maximum number of instances to test/train on  (-1 = no limit).",
            100000000, -1, Integer.MAX_VALUE);

    public IntOption timeLimitOption = new IntOption("timeLimit", 't',
            "Maximum number of seconds to test/train for (-1 = no limit).", -1,
            -1, Integer.MAX_VALUE);

    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency",
            'f',
            "How many instances between samples of the learning performance.",
            100000, 0, Integer.MAX_VALUE);

    public IntOption memCheckFrequencyOption = new IntOption(
            "memCheckFrequency", 'q',
            "How many instances between memory bound checks.", 100000, 0,
            Integer.MAX_VALUE);

    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", null, "csv", true);

    public FileOption outputPredictionFileOptionUp = new FileOption("outputPredictionFileUp", 'X',
            "File to append output predictions to.", null, "pred", true);
    
    public FileOption outputPredictionFileOptionDown = new FileOption("outputPredictionFileDown", 'Y',
            "File to append output predictions to.", null, "pred", true);
    
    public FileOption outputPredictionFileOptionMiddle = new FileOption("outputPredictionFileMiddle", 'Z',
            "File to append output predictions to.", null, "pred", true);

    //New for prequential method DEPRECATED
    public IntOption widthOption = new IntOption("width",
            'w', "Size of Window", 1000);

    public FloatOption alphaOption = new FloatOption("alpha",
            'a', "Fading factor or exponential smoothing factor", .01);
    //End New for prequential methods

    @Override
    public Class<?> getTaskResultType() {
        return LearningCurve.class;
    }

    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
        Learner learnerUp = (Learner)((Learner) getPreparedClassOption(this.learnerOption)).copy();
        Learner learnerDown = (Learner)((Learner) getPreparedClassOption(this.learnerOption)).copy();
        Learner learnerMiddle = (Learner)((Learner) getPreparedClassOption(this.learnerOption)).copy();
        ExampleStream streamUp = (ExampleStream) getPreparedClassOption(this.streamOptionUp);
        ExampleStream streamDown = (ExampleStream) getPreparedClassOption(this.streamOptionDown);
        ExampleStream streamMiddle = (ExampleStream) getPreparedClassOption(this.streamOptionMiddle);
        LearningPerformanceEvaluator evaluatorUp = (LearningPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        LearningPerformanceEvaluator evaluatorDown = (LearningPerformanceEvaluator) evaluatorUp.copy();
        LearningPerformanceEvaluator evaluatorMiddle = (LearningPerformanceEvaluator) evaluatorUp.copy();
        LearningCurve learningCurve = new LearningCurve(
                "learning evaluation instances");

        //New for prequential methods
        if (evaluatorUp instanceof WindowClassificationPerformanceEvaluator) {
            //((WindowClassificationPerformanceEvaluator) evaluator).setWindowWidth(widthOption.getValue());
            if (widthOption.getValue() != 1000) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (WindowClassificationPerformanceEvaluator -w " + widthOption.getValue() + ")");
                 return learningCurve;
            }
        }
        if (evaluatorUp instanceof EWMAClassificationPerformanceEvaluator) {
            //((EWMAClassificationPerformanceEvaluator) evaluator).setalpha(alphaOption.getValue());
            if (alphaOption.getValue() != .01) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (EWMAClassificationPerformanceEvaluator -a " + alphaOption.getValue() + ")");
                return learningCurve;
            }
        }
        if (evaluatorUp instanceof FadingFactorClassificationPerformanceEvaluator) {
            //((FadingFactorClassificationPerformanceEvaluator) evaluator).setalpha(alphaOption.getValue());
            if (alphaOption.getValue() != .01) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (FadingFactorClassificationPerformanceEvaluator -a " + alphaOption.getValue() + ")");
                return learningCurve;
            }
        }
        //End New for prequential methods

        learnerUp.setModelContext(streamUp.getHeader());
        learnerDown.setModelContext(streamDown.getHeader());
        learnerMiddle.setModelContext(streamMiddle.getHeader());
        int maxInstances = this.instanceLimitOption.getValue();
        long instancesProcessed = 0;
        int maxSeconds = this.timeLimitOption.getValue();
        int secondsElapsed = 0;
        monitor.setCurrentActivity("Evaluating learner...", -1.0);

        File dumpFile = this.dumpFileOption.getFile();
        PrintStream immediateResultStream = null;
        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile, true), true);
                } else {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFile, ex);
            }
        }
        
        //File for output predictions
        File outputPredictionFileUp = this.outputPredictionFileOptionUp.getFile();
        PrintStream outputPredictionResultStreamUp = null;
        if (outputPredictionFileUp != null) {
            try {
                if (outputPredictionFileUp.exists()) {
                	outputPredictionResultStreamUp = new PrintStream(
                            new FileOutputStream(outputPredictionFileUp, true), true);
                } else {
                	outputPredictionResultStreamUp = new PrintStream(
                            new FileOutputStream(outputPredictionFileUp), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open prediction result file: " + outputPredictionFileUp, ex);
            }
        }
        
        File outputPredictionFileDown = this.outputPredictionFileOptionDown.getFile();
        PrintStream outputPredictionResultStreamDown = null;
        if (outputPredictionFileDown != null) {
            try {
                if (outputPredictionFileDown.exists()) {
                	outputPredictionResultStreamDown = new PrintStream(
                            new FileOutputStream(outputPredictionFileDown, true), true);
                } else {
                	outputPredictionResultStreamDown = new PrintStream(
                            new FileOutputStream(outputPredictionFileDown), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open prediction result file: " + outputPredictionFileDown, ex);
            }
        }
        
        File outputPredictionFileMiddle = this.outputPredictionFileOptionMiddle.getFile();
        PrintStream outputPredictionResultStreamMiddle = null;
        if (outputPredictionFileMiddle != null) {
            try {
                if (outputPredictionFileMiddle.exists()) {
                	outputPredictionResultStreamMiddle = new PrintStream(
                            new FileOutputStream(outputPredictionFileMiddle, true), true);
                } else {
                	outputPredictionResultStreamMiddle = new PrintStream(
                            new FileOutputStream(outputPredictionFileMiddle), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open prediction result file: " + outputPredictionFileMiddle, ex);
            }
        }
        
        
        boolean firstDump = true;
        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        long lastEvaluateStartTime = evaluateStartTime;
        double RAMHoursUp = 0.0;
        double RAMHoursDown = 0.0;
        double RAMHoursMiddle = 0.0;
        while (streamUp.hasMoreInstances()
                && ((maxInstances < 0) || (instancesProcessed < maxInstances))
                && ((maxSeconds < 0) || (secondsElapsed < maxSeconds))) {
            
        	Example trainInstUp = streamUp.nextInstance();
            Example testInstUp = (Example) trainInstUp; //.copy();
            
            Example trainInstDown = streamDown.nextInstance();
            Example testInstDown = (Example) trainInstDown; //.copy();
            
            Example trainInstMiddle = streamMiddle.nextInstance();
            Example testInstMiddle = (Example) trainInstMiddle; //.copy();
            
            //testInst.setClassMissing();
            double[] predictionUp = learnerUp.getVotesForInstance(testInstUp);
            double[] predictionDown = learnerDown.getVotesForInstance(testInstDown);
            double[] predictionMiddle = learnerMiddle.getVotesForInstance(testInstMiddle);
            
            instancesProcessed++;
                                    
            // Output prediction
            if (outputPredictionFileUp != null) {
                int trueClass = (int) ((Instance) trainInstUp.getData()).classValue();                            	         
                if (predictionUp.length == 2) {                	                	
                	outputPredictionResultStreamUp.println(String.valueOf(instancesProcessed) + " P(0):" + predictionUp[0] + " " + "P(1):" + predictionUp[1] + "," +(
                            ((Instance) testInstUp.getData()).classIsMissing() == true ? " ? " : trueClass));
                }
                else if (predictionUp.length == 1) {
                	outputPredictionResultStreamUp.println(String.valueOf(instancesProcessed) + " P(0):" + predictionUp[0] + " " + "P(1):0," +(
                            ((Instance) testInstUp.getData()).classIsMissing() == true ? " ? " : trueClass));
                }  
                else {
                	outputPredictionResultStreamUp.println(String.valueOf(instancesProcessed));
                }
            }
            
            // Output prediction
            if (outputPredictionFileDown != null) {
                int trueClass = (int) ((Instance) trainInstDown.getData()).classValue();
                double doy = ((Instance)trainInstDown.getData()).value(11);
            	double h = ((Instance)trainInstDown.getData()).value(13);
            	double y = ((Instance)trainInstDown.getData()).value(15);
                if (predictionDown.length == 2) {
                	outputPredictionResultStreamDown.println(String.valueOf(instancesProcessed) + " P(0):" + predictionDown[0] + " " + "P(1):" + predictionDown[1] + "," +(
                            ((Instance) testInstDown.getData()).classIsMissing() == true ? " ? " : trueClass));
                }
                else if (predictionDown.length == 1) {
                	outputPredictionResultStreamDown.println(String.valueOf(instancesProcessed) + " P(0):" + predictionDown[0] + " " + "P(1):0," +(
                            ((Instance) testInstDown.getData()).classIsMissing() == true ? " ? " : trueClass));
                }
                else {
                	outputPredictionResultStreamDown.println(String.valueOf(instancesProcessed));
                }
            }
            
            // Output prediction
            if (outputPredictionFileMiddle != null) {
                int trueClass = (int) ((Instance) trainInstMiddle.getData()).classValue();
                double doy = ((Instance)trainInstMiddle.getData()).value(11);
            	double h = ((Instance)trainInstMiddle.getData()).value(13);
            	double y = ((Instance)trainInstMiddle.getData()).value(15);
                if (predictionMiddle.length == 2) {
                	outputPredictionResultStreamMiddle.println(String.valueOf(instancesProcessed) + " P(0):" + predictionMiddle[0] + " " + "P(1):" + predictionMiddle[1] + "," +(
                            ((Instance) testInstMiddle.getData()).classIsMissing() == true ? " ? " : trueClass));
                }
                else if (predictionMiddle.length == 1) {
                	outputPredictionResultStreamMiddle.println(String.valueOf(instancesProcessed) + " P(0):" + predictionMiddle[0] + " " + "P(1):0," +(
                            ((Instance) testInstMiddle.getData()).classIsMissing() == true ? " ? " : trueClass));
                } 
                else {
                	outputPredictionResultStreamMiddle.println(String.valueOf(instancesProcessed));
                }
            }

            //evaluator.addClassificationAttempt(trueClass, prediction, testInst.weight());
            evaluatorUp.addResult(testInstUp, predictionUp);
            evaluatorDown.addResult(testInstDown, predictionDown);
            evaluatorMiddle.addResult(testInstMiddle, predictionMiddle);
            
            learnerUp.trainOnInstance(trainInstUp);
            learnerDown.trainOnInstance(trainInstDown);
            learnerMiddle.trainOnInstance(trainInstMiddle);
            
            
            if (instancesProcessed % this.sampleFrequencyOption.getValue() == 0
                    || streamUp.hasMoreInstances() == false) {
                long evaluateTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                double time = TimingUtils.nanoTimeToSeconds(evaluateTime - evaluateStartTime);
                double timeIncrement = TimingUtils.nanoTimeToSeconds(evaluateTime - lastEvaluateStartTime);
                
                /*
                double RAMHoursIncrementUp = learnerUp.measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
                RAMHoursIncrementUp *= (timeIncrement / 3600.0); //Hours
                RAMHoursUp += RAMHoursIncrementUp;
                double RAMHoursIncrementDown = learnerDown.measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
                RAMHoursIncrementDown *= (timeIncrement / 3600.0); //Hours
                RAMHoursDown += RAMHoursIncrementDown;
                double RAMHoursIncrementMiddle = learnerMiddle.measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
                RAMHoursIncrementMiddle *= (timeIncrement / 3600.0); //Hours
                RAMHoursMiddle += RAMHoursIncrementMiddle;
                */
                
                lastEvaluateStartTime = evaluateTime;
                learningCurve.insertEntry(new LearningEvaluation(
                        new Measurement[]{
                            new Measurement(
                            "learning evaluation instances",
                            instancesProcessed),
                            new Measurement(
                            "evaluation time ("
                            + (preciseCPUTiming ? "cpu "
                            : "") + "seconds)",
                            time),
                            new Measurement(
                            "model UP cost (RAM-Hours)",
                            RAMHoursUp),
                            new Measurement(
                            "model Down cost (RAM-Hours)",
                            RAMHoursDown),
                            new Measurement(
                            "model Middle cost (RAM-Hours)",
                            RAMHoursMiddle)                             
                        },
                        evaluatorUp,learnerUp,evaluatorDown, learnerDown,evaluatorMiddle, learnerMiddle));                                
				
                if (immediateResultStream != null) {
                    if (firstDump) {
                        immediateResultStream.println(learningCurve.headerToString());
                        firstDump = false;
                    }
                    immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                    immediateResultStream.flush();
                }
            }
            if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                if (monitor.taskShouldAbort()) {
                    return null;
                }
                long estimatedRemainingInstances = streamUp.estimatedRemainingInstances();
                if (maxInstances > 0) {
                    long maxRemaining = maxInstances - instancesProcessed;
                    if ((estimatedRemainingInstances < 0)
                            || (maxRemaining < estimatedRemainingInstances)) {
                        estimatedRemainingInstances = maxRemaining;
                    }
                }
                monitor.setCurrentActivityFractionComplete(estimatedRemainingInstances < 0 ? -1.0
                        : (double) instancesProcessed
                        / (double) (instancesProcessed + estimatedRemainingInstances));
                if (monitor.resultPreviewRequested()) {
                    monitor.setLatestResultPreview(learningCurve.copy());
                }
                secondsElapsed = (int) TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()
                        - evaluateStartTime);
            }
        }
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }
        
        if (outputPredictionResultStreamUp != null) {
        	outputPredictionResultStreamUp.close();
        }
        if (outputPredictionResultStreamDown != null) {
        	outputPredictionResultStreamDown.close();
        }
        if (outputPredictionResultStreamMiddle != null) {
        	outputPredictionResultStreamMiddle.close();
        }
        
        return learningCurve;
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == EvaluatePrequentialOvA.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }
}
