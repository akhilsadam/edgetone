/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include <jive_layouts/jive_layouts.h>
// #include <juce_audio_processors/juce_audio_processors.h>
// #include <juce_dsp/juce_dsp.h>
// // #include <juce_audio_basics/juce_audio_basics.h>
// #include <juce_audio_basics/buffers/juce_AudioChannelSet.h>

#include "DemoUtilities.h"
#include "SineOscillator.h"

//==============================================================================
/**
*/

//==============================================================================
class edgetoneAudioProcessor final : public juce::AudioProcessor
{
public:
    enum
    {
        maxMidiChannel    = 16,
        maxNumberOfVoices = 5
    };
    //==============================================================================
    edgetoneAudioProcessor()
        : juce::AudioProcessor (BusesProperties()
                          .withOutput ("Output #1",  juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output #2",  juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #3",  juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #4",  juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #5",  juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #6",  juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #7",  juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #8",  juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #9",  juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #10", juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #11", juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #12", juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #13", juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #14", juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #15", juce::AudioChannelSet::stereo(), false)
                          .withOutput ("Output #16", juce::AudioChannelSet::stereo(), false))
    {
        // initialize other stuff (not related to buses)
        formatManager.registerBasicFormats();

        // for (int midiChannel = 0; midiChannel < maxMidiChannel; ++midiChannel)
        // {
        //     synth.add(new juce::Synthesiser());

        //     for (int i = 0; i < maxNumberOfVoices; ++i)
        //         synth[midiChannel]->addVoice (new juce::SamplerVoice());
        // }

        // loadNewSample (createAssetInputStream ("singing.ogg"), "ogg");
    }

    //==============================================================================
    bool canAddBus    (bool isInput) const override   { return ! isInput; }
    bool canRemoveBus (bool isInput) const override   { return ! isInput; }

    //==============================================================================
    // void prepareToPlay (double newSampleRate, int samplesPerBlock) override
    // {
    //     juce::ignoreUnused (samplesPerBlock);

    //     for (auto* s : synth)
    //         s->setCurrentPlaybackSampleRate (newSampleRate);
    // }

    void releaseResources() override {}

    // void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiBuffer) override
    // {
    //     auto busCount = getBusCount (false);

    //     for (auto busNr = 0; busNr < busCount; ++busNr)
    //     {
    //         if (synth.size() <= busNr)
    //             continue;

    //         auto midiChannelBuffer = filterMidiMessagesForChannel (midiBuffer, busNr + 1);
    //         auto audioBusBuffer = getBusBuffer (buffer, false, busNr);

    //         // Voices add to the contents of the buffer. Make sure the buffer is clear before
    //         // rendering, just in case the host left old data in the buffer.
    //         audioBusBuffer.clear();

    //         synth [busNr]->renderNextBlock (audioBusBuffer, midiChannelBuffer, 0, audioBusBuffer.getNumSamples());
    //     }
    // }

    using juce::AudioProcessor::processBlock;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor();
    bool hasEditor() const override                        { return true; }

    //==============================================================================
    const juce::String getName() const override            { return "edgetone"; }
    bool acceptsMidi() const override                      { return true; }
    bool producesMidi() const override                     { return false; }
    double getTailLengthSeconds() const override           { return 0; }
    int getNumPrograms() override                          { return 1; }
    int getCurrentProgram() override                       { return 0; }
    void setCurrentProgram (int) override                  {}
    const juce::String getProgramName (int) override       { return "None"; }
    void changeProgramName (int, const juce::String&) override   {}

    bool isBusesLayoutSupported (const BusesLayout& layout) const override
    {
        const auto& outputs = layout.outputBuses;

        return layout.inputBuses.isEmpty()
            && 1 <= outputs.size()
            && std::all_of (outputs.begin(), outputs.end(), [] (const auto& bus)
               {
                   return bus.isDisabled() || bus == AudioChannelSet::stereo();
               });
    }

    //==============================================================================
    void getStateInformation (juce::MemoryBlock&) override {}
    void setStateInformation (const void*, int) override {}

private:
    //==============================================================================
    juce::ValueTree view;
    jive::Interpreter viewInterpreter;
    static juce::MidiBuffer filterMidiMessagesForChannel (const juce::MidiBuffer& input, int channel)
    {
        juce::MidiBuffer output;

        for (const auto metadata : input)
        {
            const auto message = metadata.getMessage();

            if (message.getChannel() == channel)
                output.addEvent (message, metadata.samplePosition);
        }

        return output;
    }

    // void loadNewSample (std::unique_ptr<juce::InputStream> soundBuffer, const char* format)
    // {
    //     std::unique_ptr<juce::AudioFormatReader> formatReader (formatManager.findFormatForFileExtension (format)->createReaderFor (soundBuffer.release(), true));

    //     juce::BigInteger midiNotes;
    //     midiNotes.setRange (0, 126, true);
    //     juce::SynthesiserSound::Ptr newSound = new juce::SamplerSound ("Voice", *formatReader, midiNotes, 0x40, 0.0, 0.0, 10.0);

    //     for (auto* s : synth)
    //         s->removeSound (0);

    //     // sound = newSound;

    //     for (auto* s : synth)
    //         s->addSound (newSound);
    // }

    //==============================================================================
    juce::AudioFormatManager formatManager;
     
    float level = 0.0f;
    juce::OwnedArray<SineOscillator> oscillators;
    // juce::OwnedArray<juce::Synthesiser> synth;
    // juce::SynthesiserSound::Ptr sound;

    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (edgetoneAudioProcessor)
};