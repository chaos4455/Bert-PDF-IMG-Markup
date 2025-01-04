import os
import time
import logging
from pathlib import Path
import shutil
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from colorama import init, Fore, Style
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures
import numpy as np
from skimage import filters, morphology
from scipy.ndimage import gaussian_filter
import easyocr
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.cm as cm
import psutil
from sklearn.cluster import AgglomerativeClustering

load_dotenv()  # Carrega variáveis de ambiente do .env
init(autoreset=True)  # Inicializa o Colorama

# Configuração do Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processor_v19.log', mode='w', encoding='utf-8'),  # Log em arquivo
        logging.StreamHandler() # Log na saída do console
    ]
)

# Variáveis de Ambiente (carregadas de .env)
EXTRACTED_FOLDER = os.getenv('EXTRACTED_FOLDER', 'extracted')  # Diretório de saída
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))  # Número de threads/processos
IMAGE_FORMAT = os.getenv('IMAGE_FORMAT', 'jpeg')  # Formato da imagem de saída
IMAGE_DPI = int(os.getenv('IMAGE_DPI', '300'))  # DPI da imagem
LOG_PROCESS = os.getenv('LOG_PROCESS', 'TRUE').lower() == 'true'
REMOVE_EXTRACTED = os.getenv('REMOVE_EXTRACTED', 'TRUE').lower() == 'true'
HEATMAP_ALPHA = float(os.getenv('HEATMAP_ALPHA', '0.3')) # Transparência do heatmap
BLUR_RADIUS = int(os.getenv('BLUR_RADIUS', '10')) # Raio do Blur
THRESHOLD = float(os.getenv('THRESHOLD', '0.1')) # Limiar do Threshold
HEATMAP_COLOR_SCHEME = os.getenv('HEATMAP_COLOR_SCHEME', 'viridis') # Color Scheme do Heatmap
TEXT_HEATMAP_ALPHA = float(os.getenv('TEXT_HEATMAP_ALPHA', '0.5'))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
OCR_DETECTION_THRESHOLD = float(os.getenv('OCR_DETECTION_THRESHOLD', '0.1'))
JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', '90'))
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '1000'))


# Inicialização do OCR e do Modelo BERT
# Tenta usar detect_threshold, se falhar, usa threshold
try:
    ocr_reader = easyocr.Reader(['en'], detect_threshold=OCR_DETECTION_THRESHOLD)
except TypeError:
    try:
        ocr_reader = easyocr.Reader(['en'], threshold=OCR_DETECTION_THRESHOLD)
    except TypeError:
        ocr_reader = easyocr.Reader(['en'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Paletas de cores simplificadas
COLOR_SCHEMES = {
    'red': [(i, 0, 0) for i in range(256)],
    'green': [(0, i, 0) for i in range(256)],
    'blue': [(0, 0, i) for i in range(256)],
    'viridis': [
        (48, 7, 75), (53, 8, 87), (59, 8, 99), (65, 7, 112),
        (71, 7, 125), (77, 7, 139), (84, 9, 154), (90, 10, 170),
        (97, 12, 187), (105, 16, 203), (112, 22, 220), (120, 29, 237),
        (128, 38, 253), (136, 51, 255), (145, 65, 255), (154, 80, 254),
        (163, 97, 252), (173, 114, 249), (183, 132, 244), (193, 150, 238),
        (203, 167, 231), (213, 184, 223), (223, 199, 215), (232, 213, 206),
        (240, 227, 197), (247, 240, 188), (252, 252, 178), (255, 255, 168),
        (255, 252, 158), (255, 244, 147), (255, 233, 137), (255, 221, 126),
        (255, 208, 115), (255, 195, 104), (255, 181, 92), (255, 167, 81),
        (255, 153, 69), (255, 138, 58), (255, 123, 46), (255, 108, 35),
        (255, 92, 23), (255, 77, 12),  (255, 61, 0)
    ],
    'magma': [
        (0, 0, 0), (7, 0, 9), (13, 2, 18), (19, 5, 28),
        (24, 9, 38), (29, 13, 48), (34, 18, 58), (39, 24, 68),
        (43, 31, 78), (48, 39, 88), (52, 48, 98), (57, 58, 108),
        (61, 69, 118), (65, 81, 127), (69, 94, 137), (73, 108, 146),
        (77, 122, 154), (81, 136, 162), (85, 150, 169), (89, 164, 176),
        (92, 178, 183), (96, 192, 189), (99, 205, 195), (102, 219, 201),
        (105, 232, 206), (108, 245, 212), (111, 255, 217)
    ],
    'inferno': [
         (0, 0, 0),   (3, 1, 8),   (7, 1, 16), (11, 2, 24),
        (15, 3, 32),  (19, 4, 40),  (23, 5, 48),  (27, 7, 56),
        (32, 8, 64),  (36, 9, 72),  (40, 11, 80),  (44, 13, 88),
        (49, 15, 96), (54, 17, 104), (59, 19, 112), (64, 22, 120),
        (69, 24, 128), (75, 26, 136), (80, 29, 143), (86, 32, 151),
        (92, 35, 159), (98, 38, 166), (104, 42, 174), (111, 45, 181),
        (118, 49, 188), (125, 53, 195), (132, 58, 202), (140, 62, 209),
        (148, 67, 215), (156, 72, 222), (164, 78, 228), (172, 84, 234),
        (181, 90, 239), (190, 97, 244), (199, 104, 248), (208, 111, 252),
        (217, 119, 255), (227, 127, 255),  (236, 135, 255), (246, 145, 254),
        (255, 154, 252), (255, 164, 248), (255, 174, 243), (255, 185, 237),
        (255, 196, 230),(255, 207, 223), (255, 219, 214),(255, 231, 205),
        (255, 242, 195), (255, 253, 184),  (255, 255, 171), (255, 255, 155)
     ],
        'plasma': [
             (0, 0, 0), (3, 0, 6), (6, 1, 13), (9, 2, 19), (11, 3, 26),
        (14, 5, 32), (17, 8, 39), (20, 11, 45), (22, 15, 51), (24, 19, 57),
         (26, 23, 63), (28, 28, 69), (30, 32, 75), (31, 37, 81), (33, 42, 87),
        (34, 47, 92), (35, 53, 98), (36, 59, 104), (36, 65, 109), (37, 71, 115),
        (38, 77, 120), (38, 83, 125), (39, 90, 130), (39, 96, 135), (40, 102, 140),
         (41, 109, 144), (42, 115, 149), (44, 122, 154), (46, 128, 158), (47, 135, 162),
        (49, 142, 166), (51, 148, 170), (54, 155, 173), (56, 161, 177), (59, 168, 180),
        (61, 174, 183), (64, 181, 186), (67, 187, 189), (70, 193, 192), (73, 199, 194),
        (76, 206, 197), (79, 212, 199), (83, 218, 201), (86, 224, 203),
          (89, 230, 205), (92, 236, 208), (96, 241, 208), (100, 246, 210),
        (103, 251, 211), (106, 255, 212)
        ]
}


def list_pdf_files(root_dir='.'):
    """Lista todos os arquivos PDF em um diretório."""
    logging.info(f"🔎 Iniciando a busca por arquivos PDF em: {root_dir}")
    pdf_files = [f for f in Path(root_dir).glob('*.pdf')]
    logging.info(f"📚 Encontrados {len(pdf_files)} arquivos PDF.")
    return pdf_files

def create_dataframe(pdf_files):
    """Cria um DataFrame com informações sobre os arquivos PDF."""
    logging.info("📊 Criando DataFrame...")
    df = pd.DataFrame(pdf_files, columns=['filepath'])
    df['filename'] = df['filepath'].apply(lambda x: x.name)
    df['pages_processed'] = 0 # Coluna para contar páginas processadas
    logging.info("✅ DataFrame criado.")
    return df

def create_output_folder(filename):
    """Cria a pasta de saída para um arquivo PDF específico."""
    output_path = Path(EXTRACTED_FOLDER) / Path(filename).stem
    os.makedirs(output_path, exist_ok=True)
    return output_path

def create_image_heatmap(image_array, heatmap_type='object'):
    """Gera um mapa de calor a partir de uma matriz de imagem."""
    gray_image = np.mean(image_array, axis=2)
    
    # Aplica um filtro gaussiano
    blurred_image = gaussian_filter(gray_image, sigma=BLUR_RADIUS)
    
    # Aplica um limiar adaptativo
    thresh = filters.threshold_otsu(blurred_image)
    binary_mask = blurred_image > thresh
    
    if heatmap_type == 'object':
        # Preenche pequenos buracos e remove objetos pequenos
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=100)
        binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=50)
    elif heatmap_type == 'logo':
         binary_mask = morphology.remove_small_objects(binary_mask, min_size=50)

    # Filtra novamente
    filtered_image = gaussian_filter(binary_mask.astype(float), sigma=BLUR_RADIUS)
    
    # Escala para [0, 255]
    heatmap = (filtered_image * 255).astype(np.uint8)
    if heatmap.shape[0] == 0 or heatmap.shape[1] == 0:
         return np.zeros((image_array.shape[0],image_array.shape[1]), dtype=np.uint8)
    return heatmap


def apply_heatmap_overlay(image, heatmap, alpha=HEATMAP_ALPHA, color_scheme = HEATMAP_COLOR_SCHEME):
    """Aplica a sobreposição de mapa de calor a uma imagem."""
    
    heatmap_pil = Image.fromarray(heatmap)
    heatmap_pil = heatmap_pil.convert("RGBA")
    
    # Aplica a paleta de cores
    if color_scheme in COLOR_SCHEMES:
        colormap = COLOR_SCHEMES[color_scheme]
        # Garante que a paleta de cores tenha pelo menos 256 cores
        while len(colormap) < 256:
            colormap.extend(colormap[-1:]) # Duplica a ultima cor
    else:
        colormap = [(i, i, i) for i in range(256)] # Default grayscale if color_scheme is not found
    
    heatmap_pil = Image.new("RGBA", heatmap_pil.size)
    pixels = heatmap_pil.load()

    for y in range(heatmap_pil.size[1]):
        for x in range(heatmap_pil.size[0]):
            pixel_value = heatmap[y][x]
            pixels[x, y] = colormap[pixel_value] + (int(alpha * 255),)
            
    image.paste(heatmap_pil, (0, 0), heatmap_pil)
    return image

def extract_text_from_page(image):
    """Extrai o texto de uma imagem usando EasyOCR."""
    results = ocr_reader.readtext(np.array(image))
    return results

def get_text_embeddings(text):
    """Obtém os embeddings de texto usando o modelo BERT."""
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def create_text_heatmap(image, text_boxes, similarity_threshold = SIMILARITY_THRESHOLD, text_heatmap_alpha = TEXT_HEATMAP_ALPHA):
     """Cria um mapa de calor de texto sobrepondo na imagem."""
     image_with_text_heatmap = image.copy().convert("RGBA")
     draw = ImageDraw.Draw(image_with_text_heatmap)
     
     if not text_boxes or len(text_boxes) < 2:
         return image_with_text_heatmap
     
     texts = [text for (_, text, _) in text_boxes]
     embeddings = [get_text_embeddings(text) for text in texts]
     
     # Calcula a similaridade entre todos os pares de textos
     similarity_matrix = cosine_similarity(embeddings)
     
     #Normalização das similaridades para o range [0, 1]
     min_similarity = np.min(similarity_matrix)
     max_similarity = np.max(similarity_matrix)
     
     normalized_similarity_matrix = (similarity_matrix - min_similarity) / (max_similarity - min_similarity)
     
     # Define uma paleta de cores
     cmap = cm.get_cmap('viridis')
     
     for i, (bbox, text, _) in enumerate(text_boxes):
            x1, y1 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[2][0]), int(bbox[2][1])
             
             # Calcula a similaridade com outros textos e cria a média
            average_similarity = np.mean([normalized_similarity_matrix[i][j] for j in range(len(texts)) if i != j])
            
            if average_similarity > similarity_threshold:
                color = tuple(int(c * 255) for c in cmap(average_similarity)[:3]) + (int(text_heatmap_alpha * 255),)
                draw.rectangle([(x1, y1), (x2, y2)], fill=color)
     return image_with_text_heatmap

def resize_image(image, max_size):
    """Redimensiona a imagem mantendo a proporção."""
    
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

def cluster_text_by_similarity(image, text_boxes, num_clusters=3):
    """Agrupa os blocos de texto por similaridade semântica."""
    if not text_boxes or len(text_boxes) < 2:
        return  image
    
    texts = [text for (_, text, _) in text_boxes]
    embeddings = [get_text_embeddings(text) for text in texts]
    
    if len(embeddings) < num_clusters:
      num_clusters = len(embeddings)
    
    # Agrupar embeddings
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    clustering.fit(embeddings)
    labels = clustering.labels_
    
    # Define uma paleta de cores
    cmap = cm.get_cmap('viridis', num_clusters)
    
    image_with_text_clusters = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(image_with_text_clusters)
    
    for i, (bbox, _, _) in enumerate(text_boxes):
      x1, y1 = int(bbox[0][0]), int(bbox[0][1])
      x2, y2 = int(bbox[2][0]), int(bbox[2][1])

      color = tuple(int(c * 255) for c in cmap(labels[i])[:3]) + (int(TEXT_HEATMAP_ALPHA * 255),)
      draw.rectangle([(x1, y1), (x2, y2)], fill=color)

    return image_with_text_clusters

def process_image(image, console, page_number):
    """Processa a imagem, aplicando todos os filtros e análises."""
    
    # Redimensionar a imagem
    image = resize_image(image, MAX_IMAGE_SIZE)
    image_array = np.array(image)

    # Criar uma imagem transparente para sobreposições
    image_rgba = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # 1. Criar Mapa de Calor para Objetos (Vermelho)
    object_heatmap = create_image_heatmap(image_array, heatmap_type='object')
    image_rgba = apply_heatmap_overlay(image_rgba, object_heatmap, color_scheme='red')
    
    # 2. Criar Mapa de Calor para Blocos de Texto (Azul)
    text_heatmap = create_image_heatmap(image_array, heatmap_type='text')
    image_rgba = apply_heatmap_overlay(image_rgba, text_heatmap, color_scheme='blue')

    # 3. Extrair e Analisar o Texto
    text_boxes = extract_text_from_page(image)
    
    # 4. Agrupar texto por similaridade
    image_with_semantic_overlay = cluster_text_by_similarity(image_rgba, text_boxes)
   
    # 5. Criar Mapa de Calor para Logos (Verde)
    logo_heatmap = create_image_heatmap(image_array, heatmap_type='logo')
    final_image = apply_heatmap_overlay(image_with_semantic_overlay, logo_heatmap, color_scheme='green')
    
    # Sobrepõe a imagem com os heatmaps
    image.paste(final_image, (0, 0), final_image)


    return image

def process_pdf_page(page, output_path, page_number, filename, console):
    """Processa uma única página do PDF."""
    try:
        pix = page.get_pixmap(dpi=IMAGE_DPI)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        processed_image = process_image(image, console, page_number)
        image_path = output_path / f"page_{page_number + 1}.{IMAGE_FORMAT}"
        # Convert to RGB before saving as JPEG
        processed_image.convert('RGB').save(image_path, quality=JPEG_QUALITY, format=IMAGE_FORMAT)
        if LOG_PROCESS:
            console.log(f"🖼️  {Fore.GREEN}Página {page_number + 1} de '{filename}' salva em: {image_path}{Style.RESET_ALL}")
        return True  # Marca sucesso
    except Exception as e:
        logging.error(f"❌ Erro ao processar a página {page_number+1} de '{filename}': {e}")
        return False  # Marca falha

def process_pdf(row):
    """Processa um único arquivo PDF."""
    console = Console()  # Cria console por thread
    filepath = row['filepath']
    filename = row['filename']
    output_path = create_output_folder(filename)
    total_pages = 0  # Para marcar no DF
    pages_processed = 0  # Para rastrear páginas processadas
    try:
        doc = fitz.open(str(filepath))
        total_pages = len(doc)
        with Progress(console=console, transient=True) as progress:
            task = progress.add_task(f"⚙️  Processando '{filename}'...", total=total_pages)
            for page_number, page in enumerate(doc):
                if process_pdf_page(page, output_path, page_number, filename, console):
                    pages_processed += 1
                progress.update(task, advance=1)  # Atualiza barra de progresso
        doc.close()
        if LOG_PROCESS:
            console.log(f"✅ {Fore.BLUE}'{filename}' concluído. {pages_processed} de {total_pages} páginas processadas.{Style.RESET_ALL}")

    except Exception as e:
        logging.error(f"🚨 Erro ao processar o PDF '{filename}': {e}")
    return pages_processed

def parallel_pdf_processing(df):
    """Processa os PDFs em paralelo, atualizando o DataFrame."""
    logging.info(f"🚀 Iniciando o processamento paralelo com {MAX_WORKERS} threads.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_pdf, row) for index, row in df.iterrows()]

        # Usar tqdm para acompanhar o progresso geral
        with tqdm(total=len(futures), desc="📈 Progresso Geral", unit=" PDF") as pbar:
            for index, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    pages_processed = future.result()
                    df.loc[index, 'pages_processed'] = pages_processed
                except Exception as e:
                    logging.error(f"❌ Erro no processamento do PDF: {e}")

                pbar.update(1)  # Atualiza barra de progresso geral

    logging.info("🏁 Processamento paralelo concluído.")
    return df

def display_summary(df):
    """Exibe um resumo do processamento em tabela."""
    console = Console()
    console.print("\n[bold blue]📊 Resumo do Processamento:[/bold blue]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Arquivo", style="dim", width=40)
    table.add_column("Páginas Processadas", justify="right")

    for _, row in df.iterrows():
        table.add_row(row['filename'], str(row['pages_processed']))

    console.print(table)

def clean_extracted_folders():
    if REMOVE_EXTRACTED:
        logging.info(f"🧹 Limpando pasta de extração: {EXTRACTED_FOLDER}")
        shutil.rmtree(EXTRACTED_FOLDER, ignore_errors=True)
        logging.info("✅ Pasta de extração limpa.")
    else:
        logging.info("⚠️ Remoção da pasta de extração desabilitada")

def get_resource_usage():
    """Obtém o uso de CPU e memória."""
    cpu_percent = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    return cpu_percent, memory_usage

def display_initialization_info(console):
    """Exibe informações de inicialização no console."""
    console.print(Panel(f"[bold green]🚀 Iniciando o Processamento de PDFs 🚀[/bold green]", style="bold green"))
    console.print(f"📚 [bold blue]Bibliotecas carregadas:[/bold blue]")
    console.print(f"   - [green]fitz[/green] 📄 (PyMuPDF)")
    console.print(f"   - [green]Pillow[/green] 🖼️ (PIL)")
    console.print(f"   - [green]rich[/green] 🎨 (Console Ricos)")
    console.print(f"   - [green]colorama[/green] 🌈 (Cores no Terminal)")
    console.print(f"   - [green]tqdm[/green] ⏳ (Barra de Progresso)")
    console.print(f"   - [green]python-dotenv[/green] ⚙️ (Variáveis de Ambiente)")
    console.print(f"   - [green]scikit-image[/green] 🔬 (Processamento de Imagem)")
    console.print(f"   - [green]numpy[/green] 🔢 (Arrays Numéricos)")
    console.print(f"   - [green]scipy[/green] 📈 (Filtro Gaussiano)")
    console.print(f"   - [green]easyocr[/green] 👁️ (OCR)")
    console.print(f"   - [green]transformers[/green] 🤖 (Modelos NLP)")
    console.print(f"   - [green]torch[/green] 🔥 (Tensor Engine)")
    console.print(f"   - [green]psutil[/green] ⚙️ (Monitor de Recursos)")
    
    console.print(f"\n🤖 [bold blue]Modelos inicializados:[/bold blue]")
    console.print(f"   - [green]BERT[/green] 🧠 (Modelo de Embedding de Texto)")
    console.print(f"   - [green]EasyOCR[/green] 👁️ (Modelo de OCR)")
    
    cpu_percent, memory_usage = get_resource_usage()
    console.print(f"\n⚙️ [bold blue]Recursos:[/bold blue]")
    console.print(f"   - 🎛️ CPU: [yellow]{cpu_percent:.2f}%[/yellow]")
    console.print(f"   - 💾 Memória: [yellow]{memory_usage:.2f}%[/yellow]")

    console.print(f"\n🔀 [bold blue]Pipelines Multi-Thread:[/bold blue]")
    console.print(f"   - 🧵 Máximo de Threads: [yellow]{MAX_WORKERS}[/yellow]")
    console.print(f"   - 🔄 Processamento Paralelo Ativo")

def main():
    """Função principal que coordena todo o processo."""
    start_time = time.time()

    console = Console()
    display_initialization_info(console)

    clean_extracted_folders()  # Limpa a pasta de extração

    pdf_files = list_pdf_files()
    if not pdf_files:
        console.print(Panel("❌ Nenhum arquivo PDF encontrado.", style="bold red"))
        logging.warning("Nenhum arquivo PDF encontrado")
        return

    df = create_dataframe(pdf_files)
    df = parallel_pdf_processing(df)
    display_summary(df)

    end_time = time.time()
    duration = end_time - start_time
    console.print(Panel(f"🎉 Processo Concluído em {duration:.2f} segundos! 🎉", style="bold green"))
    logging.info(f"Tempo total de execução: {duration:.2f} segundos")
    cpu_percent, memory_usage = get_resource_usage()
    console.print(f"\n⚙️ [bold blue]Consumo Final de Recursos:[/bold blue]")
    console.print(f"   - 🎛️ CPU: [yellow]{cpu_percent:.2f}%[/yellow]")
    console.print(f"   - 💾 Memória: [yellow]{memory_usage:.2f}%[/yellow]")

if __name__ == "__main__":
    main()
