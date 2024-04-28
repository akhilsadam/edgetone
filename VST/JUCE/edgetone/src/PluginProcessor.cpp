
#include <JuceHeader.h>
#include <jive_layouts/jive_layouts.h>
// #include <juce_audio_processors/juce_audio_processors.h>
// #include <juce_dsp/juce_dsp.h>
// // #include <juce_audio_basics/juce_audio_basics.h>
// #include <juce_audio_basics/buffers/juce_AudioChannelSet.h>
#include "PluginProcessor.h"
// #include "PluginEditor.h"

juce::AudioProcessor* createPluginFilter()
{
    return new edgetoneAudioProcessor();
}
juce::AudioProcessorEditor* edgetoneAudioProcessor::createEditor() 
{

    static constexpr auto style = [] {
        return new jive::Object{
            { "background", "#14181D" },
            { "foreground", "#CDD9E5" },
            { "font-family", "Verdana" },
            { "font-size", 15 },
            {
                "#greeting",
                new jive::Object{
                    { "font-size", 25 },
                    { "foreground", "#ffffff" },
                },
            }
        };
    };

    view = //topLevel("Hello, World!");
    // juce::ValueTree {
    //               "Text",
    //               {
    //                   { "text", "Hello" },
    //               },
    //           };
    juce::ValueTree {
          "Window", // Change this to "Editor" for plugin projects
          {
              { "width", 640 },
              { "height", 400 },
              { "style", style() },
              { "justify-content", "centre" }, // Center on the main-axis (vertically)
              { "align-items", "centre" },     // Center on the cross-axis (horizontally)
          },
          {
            juce::ValueTree {
                    "Component",
                    {
                        { "align-items", "flex-start" },
                        { "width", 640 },
                        { "height", 400 },
                    },
                    {
                        juce::ValueTree{
                            "Component",
                            {
                                { "justify-content", "space-between" },
                                { "flex-direction", "row" },
                            },
                            {
                                juce::ValueTree{
                                    "Button",
                                    {
                                        { "flex-direction", "row" },
                                        { "justify-content", "flex-start" },
                                        { "margin", "0 0 5 0" },
                                        { "padding", 0 },
                                    },
                                    {
                                        juce::ValueTree{
                                            "svg",
                                            {
                                                { "viewBox", "0 0 48 48" },
                                                { "height", "100%" },
                                            },
                                            {
                                                juce::ValueTree{ "rect", { { "width", 48 }, { "height", 48 } } },
                                            },
                                        },
                                        juce::ValueTree{
                                            "Text",
                                            {
                                                { "id", "greeting" },
                                                { "text", "Back" },
                                                { "margin", "0 0 0 10" },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
          },
      };

    // // interpret() needs a juce::AudioProcessor* when interpreting "Editor"
    // // types in order to construct the juce::AudioProcessorEditor
    // if (auto editor = viewInterpreter.interpret(view, this))
    // {
    //     // When interpreting an "Editor" type, the top-level item will be a
    //     // jive::GuiItem AND a juce::AudioProcessorEditor. So we can do a
    //     // dynamic-cast here to check that the editor was created successfully.
    //     if (dynamic_cast<juce::AudioProcessorEditor*>(editor.get()))
    //     {
    //         viewInterpreter.listenTo(*editor);

    //         // Release ownership to the caller.
    //         return dynamic_cast<juce::AudioProcessorEditor*>(editor.release());
    //     }
    // }

    // // Fallback in case the editor wasn't constructed for some reason
    // return new juce::GenericAudioProcessorEditor{ *this };

    // If you're 100% sure your interpreted view is correct, you could just do:
    auto interpreted = viewInterpreter.interpret(view, this);

    return dynamic_cast<juce::AudioProcessorEditor*>(interpreted.release());
}