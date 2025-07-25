import { InputForm } from "./InputForm";

interface WelcomeScreenProps {
  handleSubmit: (
    submittedInputValue: string,
    effort: string,
    model: string
  ) => void;
  onCancel: () => void;
  isLoading: boolean;
  model: string;
  setModel: (model: string) => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  handleSubmit,
  onCancel,
  isLoading,
  model,
  setModel,
}) => (
  <div className="h-full flex flex-col items-center justify-center text-center px-4 flex-1 w-full max-w-3xl mx-auto gap-4">
    <div>
      <h1 className="text-5xl md:text-6xl font-semibold text-neutral-100 mb-3">
        Welcome.
      </h1>
      <p className="text-xl md:text-2xl text-neutral-400">
        How can I help you today?
      </p>
    </div>
    <div className="w-full mt-4">
      <InputForm
        onSubmit={handleSubmit}
        isLoading={isLoading}
        onCancel={onCancel}
        hasHistory={false}
        model={model}
        setModel={setModel}
      />
    </div>
    <p className="text-xs text-neutral-500">
      Powered by LM Studio (Qwen) and LangChain LangGraph.
    </p>
  </div>
);
