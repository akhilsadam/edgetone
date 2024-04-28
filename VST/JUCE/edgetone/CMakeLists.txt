# 3.24.1 is bundled in Visual Studio 2022 v17.4
# 3.24.1 is also bundled in CLion as of 2023
cmake_minimum_required(VERSION 3.24.1)

# on windows, we expect a unix-like make installation.
# don't forget to use "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64  to set up env on windows.
# Paths
set(ICON_PATH "icon.svg") # icon for the standalone app

#### Parameters ####
set(PRODUCT_NAME "edgetone") # JUCE's PRODUCT_NAME can have spaces
set(COMPANY_NAME "sakura-yu") # Used for the MacOS bundle name and Installers
set(COMPANY_ID "ZKyu") # A four-character plugin id, first character MUST be uppercase for AU formats
set(BUNDLE_ID "com.sakura-yu.edgetone") # Used for the MacOS bundle identifier (and signing)
set(PLUG_CODE "Z001") # A unique four-character plugin id. Note: this must have at least one upper-case character
set(FORMATS Standalone VST3) # Build plugin formats from: AAX Unity VST VST3 AU AUv3 Standalone
####################

####################
include(CPM.cmake)
include(FetchContent) # fetch all items
# This adds the melatonin blur, inspector modules and cmake includes

FetchContent_Declare(cmake-includes
    GIT_REPOSITORY https://github.com/sudara/cmake-includes.git
    GIT_TAG origin/main
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cmake-includes)
FetchContent_MakeAvailable(cmake-includes)
####################
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/cmake-includes") # load Pamplejuce CMake
include(PamplejuceVersion)

# This is the internal name of the project and the name of JUCE's shared code target
# Note: This cannot have spaces (it may be 2024, but you can't have it all!)
set(PROJECT_NAME "${PRODUCT_NAME}_v${MAJOR_VERSION}")

# Configures universal binaries and decides which version of macOS to support
# include(PamplejuceMacOS)

project(${PROJECT_NAME} VERSION ${CURRENT_VERSION}) # For simplicity, the name of the CMake project is also the name of the target
include(JUCEDefaults) # Couple tweaks from cmake includes
# include_directories(${JUCE_INCLUDE_DIRS})
# JUCE is setup as a submodule in the /JUCE folder
# Locally, you must run `git submodule update --init --recursive` once
# and later `git submodule update --remote --merge` to keep it up to date
# On Github Actions, this is done as a part of actions/checkout
# add_subdirectory(${JUCE_PATH} ${JUCE_BUILD_PATH})
# include(${JUCE_PATH}/CMakeLists.txt)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/JUCE")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/JuceLibraryCode)
# Add any other modules you want modules here, before the juce_add_plugin call
# juce_add_module(modules/my_module)

FetchContent_Declare(melatonin_blur
    GIT_REPOSITORY https://github.com/sudara/melatonin_blur.git
    GIT_TAG origin/main
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/melatonin_blur)
# FetchContent_MakeAvailable(melatonin_blur)

FetchContent_Declare(melatonin_inspector
    GIT_REPOSITORY https://github.com/sudara/melatonin_inspector.git
    GIT_TAG origin/main
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/melatonin_inspector)
# FetchContent_MakeAvailable(melatonin_inspector)


juce_add_module(${CMAKE_CURRENT_BINARY_DIR}/melatonin_inspector)
juce_add_module(${CMAKE_CURRENT_BINARY_DIR}/melatonin_blur)

CPMAddPackage("gh:ImJimmi/JIVE@1.1.1")
# See `docs/CMake API.md` in the JUCE repo for all config options
juce_add_plugin("${PROJECT_NAME}"
    # ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/${ICON_PATH}"
    COMPANY_NAME "${COMPANY_NAME}"
    BUNDLE_ID "${BUNDLE_ID}"
    COPY_PLUGIN_AFTER_BUILD FALSE # May cause permission error if True
    PLUGIN_MANUFACTURER_CODE "${COMPANY_ID}"
    PLUGIN_CODE "${PLUG_CODE}"
    FORMATS "${FORMATS}"
    PRODUCT_NAME "${PRODUCT_NAME}")

# This lets us use our code in both the JUCE targets and our Test target
# Without running into ODR violations
add_library(SharedCode INTERFACE)

# C++20, please
# Use cxx_std_23 for C++23 (as of CMake v 3.20)
target_compile_features(SharedCode INTERFACE cxx_std_20)

# Manually list all .h and .cpp files for the plugin
# If you are like me, you'll use globs for your sanity.
# Just ensure you employ CONFIGURE_DEPENDS so the build system picks up changes
# If you want to appease the CMake gods and avoid globs, manually add files like so:
# set(SourceFiles Source/PluginEditor.h Source/PluginProcessor.h Source/PluginEditor.cpp Source/PluginProcessor.cpp)
file(GLOB_RECURSE SourceFiles CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
target_sources(SharedCode INTERFACE ${SourceFiles})

# Adds a BinaryData target for embedding assets into the binary
# include(Assets)

# # MacOS only: Cleans up folder and target organization on Xcode.
# include(XcodePrettify)

# This is where you can set preprocessor definitions for JUCE and your plugin
target_compile_definitions(SharedCode
    INTERFACE

    # JUCE_WEB_BROWSER and JUCE_USE_CURL off by default
    JUCE_WEB_BROWSER=0  # If you set this to 1, add `NEEDS_WEB_BROWSER TRUE` to the `juce_add_plugin` call
    JUCE_USE_CURL=0     # If you set this to 1, add `NEEDS_CURL TRUE` to the `juce_add_plugin` call
    JUCE_VST3_CAN_REPLACE_VST2=0

    # Uncomment if you are paying for a an Indie/Pro license or releasing under GPLv3
    # JUCE_DISPLAY_SPLASH_SCREEN=0

    # lets the app known if we're Debug or Release
    CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    VERSION="${CURRENT_VERSION}"

    JIVE_GUI_ITEMS_HAVE_STYLE_SHEETS=1
    # JucePlugin_Name is for some reason doesn't use the nicer PRODUCT_NAME
    PRODUCT_NAME_WITHOUT_VERSION="Pamplejuce"
)

# Link to any other modules you added (with juce_add_module) here!
# Usually JUCE modules must have PRIVATE visibility
# See https://github.com/juce-framework/JUCE/blob/master/docs/CMake%20API.md#juce_add_module
# However, with Pamplejuce, you'll link modules to SharedCode with INTERFACE visibility
# This allows the JUCE plugin targets and the Tests target to link against it
target_link_libraries(SharedCode
    INTERFACE
    # Assets
    melatonin_blur
    melatonin_inspector
    juce_audio_plugin_client
    juce_audio_processors
    juce_audio_basics
    juce_audio_utils
    juce_audio_formats
    juce_audio_devices
    juce_dsp
    juce_gui_extra
    juce_gui_basics
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
    jive::jive_layouts
    jive::jive_style_sheets)

juce_generate_juce_header("${PROJECT_NAME}")

# Link the JUCE plugin targets our SharedCode target
target_link_libraries("${PROJECT_NAME}" PRIVATE SharedCode)

# When present, use Intel IPP for performance on Windows
if(WIN32) # Can't use MSVC here, as it won't catch Clang on Windows
    find_package(IPP)
    # target_link_libraries("${PROJECT_NAME}" PUBLIC c:/programdata/chocolatey/lib/mingw/tools/install/mingw64/x86_64-w64-mingw32/lib/libmingwex.a)
    if(IPP_FOUND)
        target_link_libraries("${PROJECT_NAME}" PUBLIC IPP::ipps IPP::ippcore IPP::ippi IPP::ippcv)
        message("IPP LIBRARIES FOUND")
        target_compile_definitions("${PROJECT_NAME}" PUBLIC PAMPLEJUCE_IPP=1)
    else()
        message("IPP LIBRARIES *NOT* FOUND")
    endif()
endif()