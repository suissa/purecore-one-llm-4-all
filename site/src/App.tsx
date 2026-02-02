import { motion, useScroll, useTransform } from 'framer-motion';
import { Terminal, Cpu, Box, ArrowRight, Github, Copy } from "lucide-react";
import { useState } from 'react';

function App() {
  const { scrollYProgress } = useScroll();
  const y = useTransform(scrollYProgress, [0, 1], ['0%', '20%']);
  
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText('bun add @purecore/one-llm-4-all');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-dark-900 text-white selection:bg-primary-500/30 selection:text-primary-100 font-sans">
      {/* Background Ambience */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="blob bg-primary-500 w-96 h-96 top-0 -left-20" />
        <div className="blob bg-indigo-500 w-96 h-96 bottom-0 -right-20 animation-delay-2000" />
        <div className="blob bg-purple-500 w-64 h-64 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 opacity-10" />
      </div>

      {/* Navbar */}
      <nav className="fixed top-0 w-full z-50 glass border-b-0 border-white/5">
        <div className="container mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-bold text-xl tracking-tight">
              one-llm-4-all
            </span>
          </div>
          <div className="flex items-center gap-6">
            <a
              href="#features"
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Features
            </a>
            <a
              href="#docs"
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Docs
            </a>
            <a
              href="https://github.com/purecore/one-llm-4-all"
              target="_blank"
              className="p-2 hover:bg-white/10 rounded-full transition-colors"
            >
              <Github size={20} />
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 pt-24 pb-20 lg:pt-48 lg:pb-32 container mx-auto px-6">
        <div className="flex flex-col lg:flex-row items-center gap-12">
          <div className="flex-1 text-center lg:text-left">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              {/* <div className="inline-flex items-center px-3 py-1 rounded-full border border-primary-500/30 bg-primary-500/10 text-primary-400 text-xs font-medium mb-6">
                <Zap size={12} className="mr-2" />
                v2.0 Released - Native Implementations
              </div> */}
              <img
                src="/logo.png"
                alt="Logo"
                className="h-[300px] w-auto object-contain mx-auto"
              />
              <h1 className="text-5xl lg:text-7xl font-bold leading-tight mb-6">
                {/* Unified AI <br /> */}
                <span className="text-gradient">Native Interface</span>
              </h1>
              <p className="text-gray-400 text-lg lg:text-xl mb-8 leading-relaxed max-w-2xl mx-auto lg:mx-0">
                A type-safe, fluent adapter for OpenAI, Anthropic, Gemini, and
                more. Running natively without external SDK dependencies.
              </p>
              <div className="flex flex-col sm:flex-row items-center gap-4 justify-center lg:justify-start">
                <button
                  onClick={copyToClipboard}
                  className="h-12 px-6 rounded-lg bg-white/5 border border-white/10 hover:border-primary-500/50 hover:bg-white/10 flex items-center gap-3 transition-all group w-full sm:w-auto cursor-pointer"
                >
                  <span className="font-mono text-sm text-gray-300">
                    $ bun add @purecore/one-llm-4-all
                  </span>
                  {copied ? (
                    <span className="text-green-400 text-xs">Copied!</span>
                  ) : (
                    <Copy
                      size={16}
                      className="text-gray-500 group-hover:text-white"
                    />
                  )}
                </button>
                <a
                  href="#docs"
                  className="h-12 px-8 rounded-lg bg-primary-600 hover:bg-primary-500 text-white font-medium flex items-center transition-all w-full sm:w-auto justify-center"
                >
                  Get Started <ArrowRight size={18} className="ml-2" />
                </a>
              </div>
            </motion.div>
          </div>

          <div className="flex-1 w-full max-w-xl">
            <motion.div
              style={{ y }}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.7, delay: 0.2 }}
              className="relative"
            >
              <div className="absolute inset-0 bg-gradient-to-tr from-primary-500/20 to-purple-500/20 rounded-2xl blur-2xl -z-10" />
              <div className="glass rounded-2xl border border-white/10 p-1 overflow-hidden">
                <div className="bg-[#0f1117] rounded-xl p-6 font-mono text-sm overflow-x-auto">
                  <div className="flex gap-2 mb-4">
                    <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/50" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500/20 border border-yellow-500/50" />
                    <div className="w-3 h-3 rounded-full bg-green-500/20 border border-green-500/50" />
                  </div>
                  <pre className="text-gray-300">
                    <code>
                      <span className="text-purple-400">import</span>{" "}
                      {`{ sendPrompt }`}{" "}
                      <span className="text-purple-400">from</span>{" "}
                      <span className="text-green-400">'one-llm-4-all'</span>;
                      <span className="text-gray-500">// Fluent Interface</span>
                      <span className="text-purple-400">const</span> response ={" "}
                      <span className="text-purple-400">await</span>{" "}
                      <span className="text-blue-400">sendPrompt</span>(
                      <span className="text-green-400">
                        'Explain quantum computing'
                      </span>
                      ,{`{`}
                      model:{" "}
                      <span className="text-green-400">'gpt-4-turbo'</span>,
                      provider: <span className="text-green-400">'openai'</span>
                      {`}`}) .<span className="text-yellow-400">getText</span>
                      ();
                      <span className="text-blue-400">console</span>
                      .log(response);
                    </code>
                  </pre>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section
        id="features"
        className="py-20 container mx-auto px-6 relative z-10"
      >
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold mb-4">Why use this?</h2>
          <p className="text-gray-400">
            Built for performance, simplicity and type-safety.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          <FeatureCard
            icon={<Cpu size={24} className="text-primary-400" />}
            title="Native Implementation"
            desc="No external SDKs. We use our own lightweight HTTP client for maximum control and minimal bundle size."
          />
          <FeatureCard
            icon={<Box size={24} className="text-indigo-400" />}
            title="Unified Types"
            desc="One interface for all providers. Switch between OpenAI, Claude, and Gemini by just changing a string."
          />
          <FeatureCard
            icon={<Terminal size={24} className="text-green-400" />}
            title="Fluent API"
            desc="Chainable methods for cleaner code. .getText(), .getJSONResponse(), .getStream() works everywhere."
          />
        </div>
      </section>

      {/* Supported Providers */}
      <section className="py-20 border-t border-white/5 bg-white/5">
        <div className="container mx-auto px-6">
          <h3 className="text-center text-sm font-semibold text-gray-500 uppercase tracking-wider mb-8">
            Supported Providers
          </h3>
          <div className="flex flex-wrap justify-center gap-12 grayscale opacity-50 hover:grayscale-0 hover:opacity-100 transition-all duration-500">
            <span className="text-2xl font-bold text-white">OpenAI</span>
            <span className="text-2xl font-bold text-white">Anthropic</span>
            <span className="text-2xl font-bold text-white">Gemini</span>
            <span className="text-2xl font-bold text-white">Groq</span>
            <span className="text-2xl font-bold text-white">Mistral</span>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-white/5">
        <div className="container mx-auto px-6 text-center text-gray-500 text-sm">
          <p>© 2026 @purecore/one-llm-4-all. MIT License.</p>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, desc }: { icon: React.ReactNode, title: string, desc: string }) {
  return (
    <motion.div 
      whileHover={{ y: -5 }}
      className="glass p-8 rounded-2xl border border-white/10 hover:border-primary-500/30 transition-colors"
    >
      <div className="w-12 h-12 rounded-lg bg-white/5 flex items-center justify-center mb-6">
        {icon}
      </div>
      <h3 className="text-xl font-semibold mb-3">{title}</h3>
      <p className="text-gray-400 leading-relaxed text-sm">{desc}</p>
    </motion.div>
  );
}

export default App;
