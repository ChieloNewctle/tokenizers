/* tslint:disable */
/* eslint-disable */
/* prettier-ignore */

/* auto-generated by NAPI-RS */

const { existsSync, readFileSync } = require('fs')
const { join } = require('path')

const { platform, arch } = process

let nativeBinding = null
let localFileExisted = false
let loadError = null

function isMusl() {
  // For Node 10
  if (!process.report || typeof process.report.getReport !== 'function') {
    try {
      const lddPath = require('child_process').execSync('which ldd').toString().trim()
      return readFileSync(lddPath, 'utf8').includes('musl')
    } catch (e) {
      return true
    }
  } else {
    const { glibcVersionRuntime } = process.report.getReport().header
    return !glibcVersionRuntime
  }
}

switch (platform) {
  case 'android':
    switch (arch) {
      case 'arm64':
        localFileExisted = existsSync(join(__dirname, 'tokenizers.android-arm64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./tokenizers.android-arm64.node')
          } else {
            nativeBinding = require('tokenizers-android-arm64')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm':
        localFileExisted = existsSync(join(__dirname, 'tokenizers.android-arm-eabi.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./tokenizers.android-arm-eabi.node')
          } else {
            nativeBinding = require('tokenizers-android-arm-eabi')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Android ${arch}`)
    }
    break
  case 'win32':
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(join(__dirname, 'tokenizers.win32-x64-msvc.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./tokenizers.win32-x64-msvc.node')
          } else {
            nativeBinding = require('tokenizers-win32-x64-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'ia32':
        localFileExisted = existsSync(join(__dirname, 'tokenizers.win32-ia32-msvc.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./tokenizers.win32-ia32-msvc.node')
          } else {
            nativeBinding = require('tokenizers-win32-ia32-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(join(__dirname, 'tokenizers.win32-arm64-msvc.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./tokenizers.win32-arm64-msvc.node')
          } else {
            nativeBinding = require('tokenizers-win32-arm64-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Windows: ${arch}`)
    }
    break
  case 'darwin':
    localFileExisted = existsSync(join(__dirname, 'tokenizers.darwin-universal.node'))
    try {
      if (localFileExisted) {
        nativeBinding = require('./tokenizers.darwin-universal.node')
      } else {
        nativeBinding = require('tokenizers-darwin-universal')
      }
      break
    } catch {}
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(join(__dirname, 'tokenizers.darwin-x64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./tokenizers.darwin-x64.node')
          } else {
            nativeBinding = require('tokenizers-darwin-x64')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(join(__dirname, 'tokenizers.darwin-arm64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./tokenizers.darwin-arm64.node')
          } else {
            nativeBinding = require('tokenizers-darwin-arm64')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on macOS: ${arch}`)
    }
    break
  case 'freebsd':
    if (arch !== 'x64') {
      throw new Error(`Unsupported architecture on FreeBSD: ${arch}`)
    }
    localFileExisted = existsSync(join(__dirname, 'tokenizers.freebsd-x64.node'))
    try {
      if (localFileExisted) {
        nativeBinding = require('./tokenizers.freebsd-x64.node')
      } else {
        nativeBinding = require('tokenizers-freebsd-x64')
      }
    } catch (e) {
      loadError = e
    }
    break
  case 'linux':
    switch (arch) {
      case 'x64':
        if (isMusl()) {
          localFileExisted = existsSync(join(__dirname, 'tokenizers.linux-x64-musl.node'))
          try {
            if (localFileExisted) {
              nativeBinding = require('./tokenizers.linux-x64-musl.node')
            } else {
              nativeBinding = require('tokenizers-linux-x64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(join(__dirname, 'tokenizers.linux-x64-gnu.node'))
          try {
            if (localFileExisted) {
              nativeBinding = require('./tokenizers.linux-x64-gnu.node')
            } else {
              nativeBinding = require('tokenizers-linux-x64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'arm64':
        if (isMusl()) {
          localFileExisted = existsSync(join(__dirname, 'tokenizers.linux-arm64-musl.node'))
          try {
            if (localFileExisted) {
              nativeBinding = require('./tokenizers.linux-arm64-musl.node')
            } else {
              nativeBinding = require('tokenizers-linux-arm64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(join(__dirname, 'tokenizers.linux-arm64-gnu.node'))
          try {
            if (localFileExisted) {
              nativeBinding = require('./tokenizers.linux-arm64-gnu.node')
            } else {
              nativeBinding = require('tokenizers-linux-arm64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'arm':
        localFileExisted = existsSync(join(__dirname, 'tokenizers.linux-arm-gnueabihf.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./tokenizers.linux-arm-gnueabihf.node')
          } else {
            nativeBinding = require('tokenizers-linux-arm-gnueabihf')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Linux: ${arch}`)
    }
    break
  default:
    throw new Error(`Unsupported OS: ${platform}, architecture: ${arch}`)
}

if (!nativeBinding) {
  if (loadError) {
    throw loadError
  }
  throw new Error(`Failed to load native binding`)
}

const {
  Decoder,
  bpeDecoder,
  byteFallbackDecoder,
  ctcDecoder,
  fuseDecoder,
  metaspaceDecoder,
  replaceDecoder,
  sequenceDecoder,
  stripDecoder,
  wordPieceDecoder,
  Encoding,
  TruncationDirection,
  TruncationStrategy,
  Model,
  BPE,
  WordPiece,
  WordLevel,
  Unigram,
  GreedyTokenizer,
  Normalizer,
  prependNormalizer,
  stripAccentsNormalizer,
  bertNormalizer,
  nfdNormalizer,
  nfkdNormalizer,
  nfcNormalizer,
  nfkcNormalizer,
  stripNormalizer,
  sequenceNormalizer,
  lowercase,
  replace,
  nmt,
  precompiled,
  JsSplitDelimiterBehavior,
  PreTokenizer,
  byteLevelPreTokenizer,
  byteLevelAlphabet,
  whitespacePreTokenizer,
  whitespaceSplitPreTokenizer,
  bertPreTokenizer,
  metaspacePreTokenizer,
  splitPreTokenizer,
  punctuationPreTokenizer,
  sequencePreTokenizer,
  charDelimiterSplit,
  digitsPreTokenizer,
  Processor,
  bertProcessing,
  robertaProcessing,
  byteLevelProcessing,
  templateProcessing,
  sequenceProcessing,
  PaddingDirection,
  AddedToken,
  Tokenizer,
  Trainer,
  slice,
  mergeEncodings,
} = nativeBinding

module.exports.Decoder = Decoder
module.exports.bpeDecoder = bpeDecoder
module.exports.byteFallbackDecoder = byteFallbackDecoder
module.exports.ctcDecoder = ctcDecoder
module.exports.fuseDecoder = fuseDecoder
module.exports.metaspaceDecoder = metaspaceDecoder
module.exports.replaceDecoder = replaceDecoder
module.exports.sequenceDecoder = sequenceDecoder
module.exports.stripDecoder = stripDecoder
module.exports.wordPieceDecoder = wordPieceDecoder
module.exports.Encoding = Encoding
module.exports.TruncationDirection = TruncationDirection
module.exports.TruncationStrategy = TruncationStrategy
module.exports.Model = Model
module.exports.BPE = BPE
module.exports.WordPiece = WordPiece
module.exports.WordLevel = WordLevel
module.exports.Unigram = Unigram
module.exports.GreedyTokenizer = GreedyTokenizer
module.exports.Normalizer = Normalizer
module.exports.prependNormalizer = prependNormalizer
module.exports.stripAccentsNormalizer = stripAccentsNormalizer
module.exports.bertNormalizer = bertNormalizer
module.exports.nfdNormalizer = nfdNormalizer
module.exports.nfkdNormalizer = nfkdNormalizer
module.exports.nfcNormalizer = nfcNormalizer
module.exports.nfkcNormalizer = nfkcNormalizer
module.exports.stripNormalizer = stripNormalizer
module.exports.sequenceNormalizer = sequenceNormalizer
module.exports.lowercase = lowercase
module.exports.replace = replace
module.exports.nmt = nmt
module.exports.precompiled = precompiled
module.exports.JsSplitDelimiterBehavior = JsSplitDelimiterBehavior
module.exports.PreTokenizer = PreTokenizer
module.exports.byteLevelPreTokenizer = byteLevelPreTokenizer
module.exports.byteLevelAlphabet = byteLevelAlphabet
module.exports.whitespacePreTokenizer = whitespacePreTokenizer
module.exports.whitespaceSplitPreTokenizer = whitespaceSplitPreTokenizer
module.exports.bertPreTokenizer = bertPreTokenizer
module.exports.metaspacePreTokenizer = metaspacePreTokenizer
module.exports.splitPreTokenizer = splitPreTokenizer
module.exports.punctuationPreTokenizer = punctuationPreTokenizer
module.exports.sequencePreTokenizer = sequencePreTokenizer
module.exports.charDelimiterSplit = charDelimiterSplit
module.exports.digitsPreTokenizer = digitsPreTokenizer
module.exports.Processor = Processor
module.exports.bertProcessing = bertProcessing
module.exports.robertaProcessing = robertaProcessing
module.exports.byteLevelProcessing = byteLevelProcessing
module.exports.templateProcessing = templateProcessing
module.exports.sequenceProcessing = sequenceProcessing
module.exports.PaddingDirection = PaddingDirection
module.exports.AddedToken = AddedToken
module.exports.Tokenizer = Tokenizer
module.exports.Trainer = Trainer
module.exports.slice = slice
module.exports.mergeEncodings = mergeEncodings
